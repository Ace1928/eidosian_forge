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
class DialogData(object):
    """
    Provides a data structure for accessing textual dialog data.

    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    ``DialogTeacher`` class.

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).

    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.

    The contents of the ``((x, y, r, c, i), new_episode?)`` tuples returned by
    the data loader is the following:

    - ``x`` (str) is a query and possibly context
    - ``y`` (iter) is an iterable of label(s) for that query
    - ``r`` (str) is the str reward for getting that query correct
    - ``c`` (iter) is an iterable of label candidates that the student can choose from
    - ``i`` (str) is a str path to an image on disk, which will be loaded by the
      data class at request-time. should always point to the raw image file.
    - ``new_episode?`` (bool) is a boolean value specifying whether that example
      is the start of a new episode. If you don't use episodes set this
      to ``True`` every time.
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        self.rank = get_rank()
        self.num_workers = num_workers()
        self.is_distributed_and_is_eval = is_distributed() and any((x in opt['datatype'] for x in ('valid', 'test', 'train:evalmode')))
        if shared:
            self.image_loader = shared.get('image_loader', None)
            self.data = shared.get('data', [])
            self.cands = shared.get('cands', None)
        else:
            self.image_loader = ImageLoader(opt)
            self.data = []
            self._load(data_loader, opt['datafile'])
            self.cands = None if cands is None else set((c for c in cands))
        self.addedCands = []
        self.copied_cands = False

    def share(self):
        """
        Share the data.
        """
        shared = {'data': self.data, 'cands': self.cands, 'image_loader': self.image_loader}
        return shared

    def _read_episode(self, data_loader):
        """
        Read one episode at a time from the provided iterable over entries.

        :param data_loader:
            an iterable which returns tuples in the format described in the
            class docstring.
        """
        episode = []
        for entry, new in data_loader:
            if new and len(episode) > 0:
                yield tuple(episode)
                episode = []
            episode.append(entry)
        if len(episode) > 0:
            yield tuple(episode)

    def _load(self, data_loader, datafile):
        """
        Load up data from an iterable over tuples described in the class docs.

        :param iter data_loader:
            an iterator which returns tuples in the format described in the
            class docstring.
        :param str datafile:
        """
        for i, episode in enumerate(self._read_episode(data_loader(datafile))):
            if not self.is_distributed_and_is_eval or i % self.num_workers == self.rank:
                self.data.append(episode)

    def num_episodes(self):
        """
        Return number of episodes in the dataset.
        """
        return len(self.data)

    def num_examples(self):
        """
        Return total number of entries available.

        Each episode has at least one entry, but might have many more.
        """
        if hasattr(self, '_num_examples_cache'):
            return self._num_examples_cache
        self._num_examples_cache = sum((len(episode) for episode in self.data))
        return self._num_examples_cache

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode. Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        if episode_idx >= len(self.data):
            return ({'episode_done': True}, True)
        next_episode_idx_for_rank = episode_idx + 1
        episode = self.data[episode_idx]
        entry = episode[entry_idx]
        episode_done = entry_idx == len(episode) - 1
        end_of_data = episode_done and next_episode_idx_for_rank >= len(self.data)
        table = self.build_table(entry)
        table['episode_done'] = episode_done
        return (table, end_of_data)

    def build_table(self, entry):
        """
        Packs an entry into an action-observation dictionary.

        :param entry: a tuple in the form described in the class docstring.
        """
        if isinstance(entry, (dict, Message)):
            if 'eval_labels' in entry or 'eval_label' in entry:
                raise KeyError('Labels are converted to eval_labels automatically. Please do not set them in setup_data.')
            if 'episode_done' in entry:
                raise KeyError("episode_done is set automatically for you. Please don't set it in setup_data.")
            if 'label' in entry:
                label = entry.pop('label')
                assert isinstance(label, str)
                entry['labels'] = (label,)
            if 'labels' in entry and isinstance(entry['labels'], str):
                entry['labels'] = (entry['labels'],)
            table = entry.copy()
        elif isinstance(entry, (Tuple, List)):
            table = {}
            if entry[0] is not None:
                table['text'] = entry[0]
            if len(entry) > 1 and entry[1] is not None:
                l = entry[1]
                if isinstance(l, str):
                    l = (l,)
                table['labels'] = l
            if len(entry) > 2 and entry[2] is not None:
                table['reward'] = entry[2]
            if len(entry) > 3 and entry[3] is not None:
                table['label_candidates'] = entry[3]
            if len(entry) > 4 and entry[4] is not None:
                img = self.image_loader.load(entry[4])
                if img is not None:
                    table['image'] = img
        else:
            raise TypeError(f'items out of setup_data should be dict, Message, list, or tuple. Got {type(entry)})')
        if table.get('labels', None) is not None and self.cands is not None:
            if self.addedCands:
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    if not self.copied_cands:
                        self.cands = self.cands.copy()
                        self.copied_cands = True
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands
        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')
        if isinstance(table, dict):
            table = Message(table)
        return table