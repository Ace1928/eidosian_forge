import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
class _InstancePrivate:
    """
    initialized: bool
        Flag that can be tested to see if e.g. constant Parameters can still be set
    parameters_state: dict
        Dict holding some transient states
    dynamic_watchers: defaultdict
        Dynamic watchers
    ref_watchers: list[Watcher]
        Watchers used for internal references
    params: dict
        Dict of parameter_name:parameter
    refs: dict
        Dict of parameter name:reference
    watchers: dict
        Dict of dict:
            parameter_name:
                parameter_attribute (e.g. 'value'): list of `Watcher`s
    values: dict
        Dict of parameter name: value
    """
    __slots__ = ['initialized', 'parameters_state', 'dynamic_watchers', 'params', 'async_refs', 'refs', 'ref_watchers', 'syncing', 'watchers', 'values', 'explicit_no_refs']

    def __init__(self, initialized=False, parameters_state=None, dynamic_watchers=None, refs=None, params=None, watchers=None, values=None, explicit_no_refs=None):
        self.initialized = initialized
        self.explicit_no_refs = [] if explicit_no_refs is None else explicit_no_refs
        self.syncing = set()
        if parameters_state is None:
            parameters_state = {'BATCH_WATCH': False, 'TRIGGER': False, 'events': [], 'watchers': []}
        self.ref_watchers = []
        self.async_refs = {}
        self.parameters_state = parameters_state
        self.dynamic_watchers = defaultdict(list) if dynamic_watchers is None else dynamic_watchers
        self.params = {} if params is None else params
        self.refs = {} if refs is None else refs
        self.watchers = {} if watchers is None else watchers
        self.values = {} if values is None else values

    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)