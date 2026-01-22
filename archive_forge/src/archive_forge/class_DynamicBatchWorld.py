import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
class DynamicBatchWorld(World):

    def __init__(self, opt: Opt, world: Union[DialogPartnerWorld, MultiWorld]):
        super().__init__(opt)
        self.opt = opt
        self.agents = []
        if isinstance(world, (BatchWorld, MultiAgentDialogWorld)):
            raise TypeError('World must be a DialogPartnerWorld or a MultiWorld of DialogPartnerWorld')
        if len(world.get_agents()) != 2:
            raise AssertionError('Dynamic batch only works in a fixed dialog world with two agents.')
        if not hasattr(world.get_model_agent(), 'batch_act'):
            raise TypeError("Model agent doesn't have batch_act.")
        self.truncate = opt.get('text_truncate', None) or opt.get('truncate', None)
        self.l_truncate = opt.get('label_truncate', None) or opt.get('truncate', None)
        if self.truncate is None or self.truncate < 0:
            raise ValueError('You must use --text-truncate or --truncate in order to use dynamic batching.')
        self._BUFFER_SIZE = 1021
        if opt['dynamic_batching'] == 'full':
            self.max_batch_size = self._BUFFER_SIZE
        else:
            self.max_batch_size = opt['batchsize']
        self.world = world
        self.max_words = (self.l_truncate + self.truncate) * opt['batchsize']
        self.worlds = [world.clone() for _ in range(self._BUFFER_SIZE)]
        self.reset()

    def reset(self):
        super().reset()
        self._task_acts = [None for _ in range(self._BUFFER_SIZE)]
        self._obs = [None for _ in range(self._BUFFER_SIZE)]
        self._scores = [None for _ in range(self._BUFFER_SIZE)]
        self.acts = [None, None]
        self.number_parleys = 0
        self.total_exs = 0
        self.world.reset()
        self.rng = random.Random(4)
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        super().reset_metrics()
        self.world.reset_metrics()
        for w in self.worlds:
            w.reset_metrics()

    def epoch_done(self):
        return self.world.epoch_done() or (all((w.epoch_done() for w in self.worlds)) and all((s is None for s in self._scores)))

    def num_examples(self):
        return self.world.num_examples()

    def num_episodes(self):
        return self.world.num_episodes()

    def _ceil(self, n):
        """
        Round to the nearest multiple of 8.

        TensorCores only work when a tensor is a multiple of 8 in almost all
        dimensions. This means all examples cost is related to their nearest
        multiple of 8.

        See https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/ for
        more information.
        """
        from parlai.utils.torch import FP16_PAD_SIZE
        return (n + FP16_PAD_SIZE - 1) // FP16_PAD_SIZE * FP16_PAD_SIZE

    def _score(self, obs):
        if 'text_vec' in obs:
            return tuple((self._ceil(len(obs[key])) for key in ['text_vec', 'labels_vec', 'eval_labels_vec'] if key in obs))
        else:
            return None

    def parley(self):
        indices = []
        for i in range(self._BUFFER_SIZE):
            if self._scores[i] is not None:
                indices.append(i)
                continue
            if self.worlds[i].epoch_done():
                continue
            if hasattr(self.world, 'parley_init'):
                self.worlds[i].parley_init()
            act = self.worlds[i].get_task_agent().act()
            self._task_acts[i] = act
            self._task_acts[i]['dyn_batch_idx'] = i
            obs = self.worlds[i].get_model_agent().observe(act)
            self._obs[i] = obs
            self._scores[i] = self._score(obs)
            if self._scores[i] is not None:
                indices.append(i)
        assert len(indices) != 0, 'DynamicBatchWorld ran out of data!'
        assert not any((self._scores[i] is None for i in indices))
        indices = sorted(indices, key=lambda i: self._scores[i] + (self.rng.random(),))
        batch = []
        width = 0
        indices_idx = random.randint(0, len(indices) - 1)
        while indices_idx < len(indices) - 1 and sum(self._scores[indices[indices_idx]]) == sum(self._scores[indices[indices_idx + 1]]):
            indices_idx += 1
        while indices:
            index = indices[indices_idx]
            this_width = self._ceil(sum(self._scores[index]))
            new_width = max(width, this_width)
            new_bsz = len(batch) + 1
            new_words = new_width * new_bsz
            if new_words <= self.max_words and new_bsz <= self.max_batch_size:
                width = new_width
                batch.append(index)
                indices.pop(indices_idx)
                indices_idx = max(indices_idx - 1, 0)
            else:
                break
        while len(batch) > 4 and len(batch) % 4 != 0:
            batch.pop(-1)
        assert self._ceil(width) * len(batch) <= self.max_words
        assert len(batch) > 0
        assert len(batch) <= self.max_batch_size
        acts = self.world.get_model_agent().batch_act([self._obs[i] for i in batch])
        self.acts = [[self._task_acts[i] for i in batch], acts]
        for i, act in zip(batch, acts):
            self.worlds[i].get_task_agent().observe(act)
            self.worlds[i].get_model_agent().self_observe(act)
            act = self.worlds[i].get_task_agent().act()
            self._task_acts[i] = act
            self._task_acts[i]['dyn_batch_idx'] = i
            obs = self.worlds[i].get_model_agent().observe(act)
            self._scores[i] = self._score(obs)
            self._obs[i] = obs
        self.total_parleys += 1
        self.total_exs += len(batch)

    def get_total_exs(self):
        return self.total_exs

    def get_total_epochs(self):
        return self.total_exs / self.num_examples()

    def report(self):
        return self.world.report()