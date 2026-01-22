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
class ParamOverrides(dict):
    """
    A dictionary that returns the attribute of a specified object if
    that attribute is not present in itself.

    Used to override the parameters of an object.
    """

    def __init__(self, overridden, dict_, allow_extra_keywords=False):
        """

        If allow_extra_keywords is False, then all keys in the
        supplied dict_ must match parameter names on the overridden
        object (otherwise a warning will be printed).

        If allow_extra_keywords is True, then any items in the
        supplied dict_ that are not also parameters of the overridden
        object will be available via the extra_keywords() method.
        """
        self._overridden = overridden
        dict.__init__(self, dict_)
        if allow_extra_keywords:
            self._extra_keywords = self._extract_extra_keywords(dict_)
        else:
            self._check_params(dict_)

    def extra_keywords(self):
        """
        Return a dictionary containing items from the originally
        supplied `dict_` whose names are not parameters of the
        overridden object.
        """
        return self._extra_keywords

    def param_keywords(self):
        """
        Return a dictionary containing items from the originally
        supplied `dict_` whose names are parameters of the
        overridden object (i.e. not extra keywords/parameters).
        """
        return {key: self[key] for key in self if key not in self.extra_keywords()}

    def __missing__(self, name):
        return getattr(self._overridden, name)

    def __repr__(self):
        return dict.__repr__(self) + ' overriding params from %s' % repr(self._overridden)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, name, val):
        if not name.startswith('_'):
            self.__setitem__(name, val)
        else:
            dict.__setattr__(self, name, val)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self.__dict__ or key in self._overridden.param

    def _check_params(self, params):
        """
        Print a warning if params contains something that is not a
        Parameter of the overridden object.
        """
        overridden_object_params = list(self._overridden.param)
        for item in params:
            if item not in overridden_object_params:
                self.param.warning("'%s' will be ignored (not a Parameter).", item)

    def _extract_extra_keywords(self, params):
        """
        Return any items in params that are not also
        parameters of the overridden object.
        """
        extra_keywords = {}
        overridden_object_params = list(self._overridden.param)
        for name, val in params.items():
            if name not in overridden_object_params:
                extra_keywords[name] = val
        return extra_keywords