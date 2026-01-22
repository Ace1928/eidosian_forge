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
class URIOpt(Opt):
    """Opt with URI type

    Option with ``type`` :class:`oslo_config.types.URI`

    :param name: the option's name
    :param max_length: If positive integer, the value must be less than or
                       equal to this parameter.
    :param schemes: list of valid URI schemes, e.g. 'https', 'ftp', 'git'
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionadded:: 3.12

    .. versionchanged:: 3.14
       Added *max_length* parameter
    .. versionchanged:: 3.18
       Added *schemes* parameter
    """

    def __init__(self, name, max_length=None, schemes=None, **kwargs):
        type = types.URI(max_length=max_length, schemes=schemes)
        super(URIOpt, self).__init__(name, type=type, **kwargs)