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
class ListOpt(Opt):
    """Option with List(String) type

    Option with ``type`` :class:`oslo_config.types.List`

    :param name: the option's name
    :param item_type: type of items (see :class:`oslo_config.types`)
    :param bounds: if True the value should be inside "[" and "]" pair
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionchanged:: 2.5
       Added *item_type* and *bounds* parameters.
    """

    def __init__(self, name, item_type=None, bounds=None, **kwargs):
        super(ListOpt, self).__init__(name, type=types.List(item_type=item_type, bounds=bounds), **kwargs)