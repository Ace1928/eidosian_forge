import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
class MultiTaskTeacher(Teacher):
    """
    MultiTaskTeacher which teaches multiple tasks.

    Creates a teacher that is actually a set of teachers each based on a task
    string -- each of these teachers will get called in turn,
    either randomly or in order.  They are all in the same world (they are the
    same agent switching tasks).

    The task string format is described for the ``create_task_agents()``
    function above.
    """

    def __init__(self, opt: Opt, shared=None):
        self.tasks: List[Agent] = []
        self.opt = opt
        self.id = opt['task']
        if shared and 'tasks' in shared:
            self.tasks = [create_agent_from_shared(t) for t in shared['tasks']]
        else:
            tasks = opt['task'].split(',')
            for k in tasks:
                k = k.strip()
                if k:
                    opt_singletask = copy.deepcopy(opt)
                    opt_singletask['task'] = k
                    self.tasks.extend(create_task_agent_from_taskname(opt_singletask))
        self.task_idx = -1
        self.new_task = True
        self.random = opt.get('datatype') == 'train'
        self.cum_task_weights = [1] * len(self.tasks)
        self.task_choices = range(len(self.tasks))
        weights = self.opt.get('multitask_weights', [1])
        if weights == 'stochastic':
            weights = [t.num_episodes() for t in self.tasks]
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight

    def num_examples(self):
        """
        Return the number of examples.
        """
        if not hasattr(self, 'num_exs'):
            tasks_num_exs = [t.num_examples() for t in self.tasks]
            if any((num is None for num in tasks_num_exs)):
                self.num_exs = None
            else:
                self.num_exs = sum(tasks_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            tasks_num_eps = [t.num_episodes() for t in self.tasks]
            if any((num is None for num in tasks_num_eps)):
                self.num_eps = None
            else:
                self.num_eps = sum(tasks_num_eps)
        return self.num_eps

    def observe(self, observation):
        """
        Make an observation.
        """
        return self.tasks[self.task_idx].observe(observation)

    def act(self):
        """
        Act on the previous observation.
        """
        if self.new_task:
            self.new_task = False
            if self.random:
                self.task_idx = random.choices(self.task_choices, cum_weights=self.cum_task_weights)[0]
            else:
                for _ in range(len(self.tasks)):
                    self.task_idx = (self.task_idx + 1) % len(self.tasks)
                    if not self.tasks[self.task_idx].epoch_done():
                        break
                if self.tasks[self.task_idx].epoch_done():
                    return {'episode_done': True}
        t = self.tasks[self.task_idx].act()
        if t['episode_done']:
            self.new_task = True
        return t

    def epoch_done(self):
        """
        Return whether all subtasks are completed.
        """
        for t in self.tasks:
            if not t.epoch_done():
                return False
        return True

    def report(self):
        """
        Report aggregated metrics across all subtasks.
        """
        return aggregate_named_reports({t.getID(): t.report() for t in self.tasks}, micro_average=self.opt.get('aggregate_micro', False))

    def reset(self):
        """
        Reset all subtasks.
        """
        for t in self.tasks:
            t.reset()

    def reset_metrics(self):
        """
        Reset metrics for each subtask.
        """
        for t in self.tasks:
            t.reset_metrics()

    def share(self):
        """
        Shares this teacher by sharing each subtask.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['tasks'] = [t.share() for t in self.tasks]
        return shared

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for t in self.tasks:
            t.shutdown()