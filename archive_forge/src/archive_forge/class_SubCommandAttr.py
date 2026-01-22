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
class SubCommandAttr:
    """Helper class.

        Represents the name and arguments of an argparse sub-parser.
        """

    def __init__(self, conf, group, dest):
        """Construct a SubCommandAttr object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            :param dest: the name of the sub-parser
            """
        self._conf = conf
        self._group = group
        self._dest = dest

    def __getattr__(self, name):
        """Look up a sub-parser name or argument value."""
        if name == 'name':
            name = self._dest
            if self._group is not None:
                name = self._group.name + '_' + name
            return getattr(self._conf._namespace, name)
        if name in self._conf:
            raise DuplicateOptError(name)
        try:
            return getattr(self._conf._namespace, name)
        except AttributeError:
            raise NoSuchOptError(name)