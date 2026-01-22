import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
def __clear_cache(f):

    @functools.wraps(f)
    def __inner(self, *args, **kwargs):
        if kwargs.pop('clear_cache', True):
            result = f(self, *args, **kwargs)
            self.__cache.clear()
            return result
        else:
            return f(self, *args, **kwargs)
    return __inner