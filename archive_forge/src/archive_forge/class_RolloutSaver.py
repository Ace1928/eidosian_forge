import collections
import copy
import gymnasium as gym
import json
import os
from pathlib import Path
import shelve
import typer
import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.common import CLIArguments as cli
from ray.train._checkpoint import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self, outfile=None, use_shelve=False, write_update_file=False, target_steps=None, target_episodes=None, save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ('__progress_' + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                self._shelf = shelve.open(self._outfile)
            else:
                try:
                    with open(self._outfile, 'wb') as _:
                        pass
                except IOError as x:
                    print('Can not open {} for writing - cancelling rollouts.'.format(self._outfile))
                    raise x
            if self._write_update_file:
                self._update_file = self._get_tmp_progress_filename().open(mode='w')
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            self._shelf['num_episodes'] = self._num_episodes
            self._shelf.close()
        elif self._outfile and (not self._use_shelve):
            cloudpickle.dump(self._rollouts, open(self._outfile, 'wb'))
        if self._update_file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return '{} / {} episodes completed'.format(self._num_episodes, self._target_episodes)
        elif self._target_steps:
            return '{} / {} steps completed'.format(self._total_steps, self._target_steps)
        else:
            return '{} episodes completed'.format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + '\n')
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, terminated, truncated, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append([obs, action, next_obs, reward, terminated, truncated, info])
            else:
                self._current_rollout.append([obs, action, next_obs, reward, terminated, truncated])
        self._total_steps += 1