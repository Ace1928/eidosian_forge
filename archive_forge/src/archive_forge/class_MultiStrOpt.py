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
class MultiStrOpt(MultiOpt):
    """MultiOpt with a MultiString ``item_type``.

    MultiOpt with a default :class:`oslo_config.types.MultiString` item
    type.

    :param name: the option's name
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`MultiOpt`
    """

    def __init__(self, name, **kwargs):
        super(MultiStrOpt, self).__init__(name, item_type=types.MultiString(), **kwargs)