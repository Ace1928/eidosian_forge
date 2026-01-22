import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def add_filtering_arguments(self, parser):
    if not self.filter_attrs:
        return
    group_parser = parser.add_argument_group('filtering arguments')
    collection = self.resource_plural or '%ss' % self.resource
    for attr in self.filter_attrs:
        if isinstance(attr, str):
            attr_name = attr
            attr_defs = self.default_attr_defs[attr]
        else:
            attr_name = attr['name']
            attr_defs = attr
        option_name = '--%s' % attr_name.replace('_', '-')
        params = attr_defs.get('argparse_kwargs', {})
        try:
            help_msg = attr_defs['help'] % collection
        except TypeError:
            help_msg = attr_defs['help']
        if attr_defs.get('boolean', False):
            add_arg_func = functools.partial(utils.add_boolean_argument, group_parser)
        else:
            add_arg_func = group_parser.add_argument
        add_arg_func(option_name, help=help_msg, **params)