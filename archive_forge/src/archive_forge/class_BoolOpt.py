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
class BoolOpt(Opt):
    """Boolean options.

    Bool opts are set to True or False on the command line using --optname or
    --nooptname respectively.

    In config files, boolean values are cast with Boolean type.

    :param name: the option's name
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`
    """

    def __init__(self, name, **kwargs):
        if 'positional' in kwargs:
            raise ValueError('positional boolean args not supported')
        super(BoolOpt, self).__init__(name, type=types.Boolean(), **kwargs)

    def _add_to_cli(self, parser, group=None):
        """Extends the base class method to add the --nooptname option."""
        super(BoolOpt, self)._add_to_cli(parser, group)
        self._add_inverse_to_argparse(parser, group)

    def _add_inverse_to_argparse(self, parser, group):
        """Add the --nooptname option to the option parser."""
        container = self._get_argparse_container(parser, group)
        kwargs = self._get_argparse_kwargs(group, action='store_false')
        prefix = self._get_argparse_prefix('no', group.name if group else None)
        deprecated_names = []
        for opt in self.deprecated_opts:
            deprecated_name = self._get_deprecated_cli_name(opt.name, opt.group, prefix='no')
            if deprecated_name is not None:
                deprecated_names.append(deprecated_name)
        kwargs['help'] = 'The inverse of --' + self.name
        self._add_to_argparse(parser, container, self.name, None, kwargs, prefix, self.positional, deprecated_names)

    def _get_argparse_kwargs(self, group, action='store_true', **kwargs):
        """Extends the base argparse keyword dict for boolean options."""
        kwargs = super(BoolOpt, self)._get_argparse_kwargs(group, **kwargs)
        if 'type' in kwargs:
            del kwargs['type']
        if 'metavar' in kwargs:
            del kwargs['metavar']
        kwargs['action'] = action
        return kwargs