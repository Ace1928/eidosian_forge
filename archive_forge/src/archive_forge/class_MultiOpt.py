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
class MultiOpt(Opt):
    """Multi-value option.

    Multi opt values are typed opts which may be specified multiple times.
    The opt value is a list containing all the values specified.

    :param name: the option's name
    :param item_type: Type of items (see :class:`oslo_config.types`)
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    For example::

       cfg.MultiOpt('foo',
                    item_type=types.Integer(),
                    default=None,
                    help="Multiple foo option")

    The command line ``--foo=1 --foo=2`` would result in ``cfg.CONF.foo``
    containing ``[1,2]``

    .. versionadded:: 1.3
    """
    multi = True

    def __init__(self, name, item_type, **kwargs):
        super(MultiOpt, self).__init__(name, item_type, **kwargs)

    def _get_argparse_kwargs(self, group, **kwargs):
        """Extends the base argparse keyword dict for multi value options."""
        kwargs = super(MultiOpt, self)._get_argparse_kwargs(group)
        if not self.positional:
            kwargs['action'] = 'append'
        else:
            kwargs['nargs'] = '*'
        return kwargs