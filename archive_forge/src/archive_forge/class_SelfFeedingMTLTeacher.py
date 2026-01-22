import copy
import json
import os
import re
import numpy as np
from parlai.core.teachers import ParlAIDialogTeacher, MultiTaskTeacher
from projects.self_feeding.utils import add_person_tokens
from .build import build
class SelfFeedingMTLTeacher(MultiTaskTeacher):
    """
    Creates a teacher that is actually a set of teachers each based on a task string--
    each of these teachers will get called in turn, either randomly or in order. They
    are all in the same world (they are the same agent switching tasks).

    More specifically, this child class of MultiTaskTeacher supports multitask learning
    with batches (ensuring that all batches only have data from a single task at a time)
    """

    def __init__(self, opt, shared=None):
        if opt['task'] == 'self_feeding:SelfFeedingMTLTeacher':
            opt = copy.deepcopy(opt)
            opt['task'] = 'self_feeding:SelfFeedingTeacher'
        super().__init__(opt, shared)
        num_batches = np.array([t.num_examples() / t.bsz for t in self.tasks])
        self.sampling_prob = num_batches / np.sum(num_batches)
        self.task_idx_assignment = -1
        self.new_task = True
        self.random = opt.get('datatype') == 'train'

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)

    def observe(self, observation):
        return self.tasks[self.task_idx].observe(observation)

    def batch_act(self, batch_observation):
        task_idx = self.get_task_index()
        actions = []
        for _ in range(self.tasks[task_idx].bsz):
            action = self.tasks[task_idx].act()
            action['subtask'] = self.tasks[task_idx].opt['subtask']
            actions.append(action)
        return actions

    def act(self):
        self.task_idx = self.get_task_index()
        if self.task_idx < 0:
            return {'episode_done': True}
        action = self.tasks[self.task_idx].act()
        action['subtask'] = self.tasks[self.task_idx].opt['subtask']
        return action

    def get_task_index(self):
        if self.opt['datatype'] == 'train':
            return np.random.choice(range(len(self.tasks)), p=self.sampling_prob)
        else:
            for i, subtask in enumerate(self.tasks):
                if not subtask.epoch_done():
                    return i
        return -1