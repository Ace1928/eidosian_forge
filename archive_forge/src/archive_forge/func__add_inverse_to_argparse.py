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