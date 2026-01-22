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
class IPOpt(Opt):
    """Opt with IPAddress type

    Option with ``type`` :class:`oslo_config.types.IPAddress`

    :param name: the option's name
    :param version: one of either ``4``, ``6``, or ``None`` to specify
       either version.
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionadded:: 1.4
    """

    def __init__(self, name, version=None, **kwargs):
        super(IPOpt, self).__init__(name, type=types.IPAddress(version), **kwargs)