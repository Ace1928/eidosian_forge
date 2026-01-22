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
class OptGroup:
    """Represents a group of opts.

    CLI opts in the group are automatically prefixed with the group name.

    Each group corresponds to a section in config files.

    An OptGroup object has no public methods, but has a number of public string
    properties:

    .. py:attribute:: name

        the name of the group

    .. py:attribute:: title

        the group title as displayed in --help

    .. py:attribute:: help

        the group description as displayed in --help

    :param name: the group name
    :type name: str
    :param title: the group title for --help
    :type title: str
    :param help: the group description for --help
    :type help: str
    :param dynamic_group_owner: The name of the option that controls
                                repeated instances of this group.
    :type dynamic_group_owner: str
    :param driver_option: The name of the option within the group that
                          controls which driver will register options.
    :type driver_option: str

    """

    def __init__(self, name, title=None, help=None, dynamic_group_owner='', driver_option=''):
        """Constructs an OptGroup object."""
        self.name = name
        self.title = '%s options' % name if title is None else title
        self.help = help
        self.dynamic_group_owner = dynamic_group_owner
        self.driver_option = driver_option
        self._opts = {}
        self._argparse_group = None
        self._driver_opts = {}

    def _save_driver_opts(self, opts):
        """Save known driver opts.

        :param opts: mapping between driver name and list of opts
        :type opts: dict

        """
        self._driver_opts.update(opts)

    def _get_generator_data(self):
        """Return a dict with data for the sample generator."""
        return {'help': self.help or '', 'dynamic_group_owner': self.dynamic_group_owner, 'driver_option': self.driver_option, 'driver_opts': self._driver_opts}

    def _register_opt(self, opt, cli=False):
        """Add an opt to this group.

        :param opt: an Opt object
        :param cli: whether this is a CLI option
        :returns: False if previously registered, True otherwise
        :raises: DuplicateOptError if a naming conflict is detected
        """
        if _is_opt_registered(self._opts, opt):
            return False
        self._opts[opt.dest] = {'opt': opt, 'cli': cli}
        return True

    def _unregister_opt(self, opt):
        """Remove an opt from this group.

        :param opt: an Opt object
        """
        if opt.dest in self._opts:
            del self._opts[opt.dest]

    def _get_argparse_group(self, parser):
        if self._argparse_group is None:
            'Build an argparse._ArgumentGroup for this group.'
            self._argparse_group = parser.add_argument_group(self.title, self.help)
        return self._argparse_group

    def _clear(self):
        """Clear this group's option parsing state."""
        self._argparse_group = None

    def __str__(self):
        return self.name