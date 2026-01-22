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
class FloatOpt(Opt):
    """Option with Float type

    Option with ``type`` :class:`oslo_config.types.Float`

    :param name: the option's name
    :param min: minimum value the float can take
    :param max: maximum value the float can take
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionchanged:: 3.14

       Added *min* and *max* parameters.
    """

    def __init__(self, name, min=None, max=None, **kwargs):
        super(FloatOpt, self).__init__(name, type=types.Float(min, max), **kwargs)