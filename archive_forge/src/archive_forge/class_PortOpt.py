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
class PortOpt(Opt):
    """Option for a TCP/IP port number.  Ports can range from 0 to 65535.

    Option with ``type`` :class:`oslo_config.types.Integer`

    :param name: the option's name
    :param min: minimum value the port can take
    :param max: maximum value the port can take
    :param choices: Optional sequence of either valid values or tuples of valid
        values with descriptions.
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionadded:: 2.6
    .. versionchanged:: 3.2
       Added *choices* parameter.
    .. versionchanged:: 3.4
       Allow port number with 0.
    .. versionchanged:: 3.16
       Added *min* and *max* parameters.
    .. versionchanged:: 5.2
       The *choices* parameter will now accept a sequence of tuples, where each
       tuple is of form (*choice*, *description*)
    """

    def __init__(self, name, min=None, max=None, choices=None, **kwargs):
        type = types.Port(min=min, max=max, choices=choices, type_name='port value')
        super(PortOpt, self).__init__(name, type=type, **kwargs)