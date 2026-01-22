import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
class SubExecutionsBaseLister(cliff_lister):

    def _get_format_function(self):
        raise NotImplementedError

    def _get_resources_function(self):
        raise NotImplementedError

    def get_parser(self, prog_name):
        parser = super(SubExecutionsBaseLister, self).get_parser(prog_name)
        parser.add_argument('id', metavar='ID', help='origin id')
        parser.add_argument('--errors-only', dest='errors_only', action='store_true', help='Only error paths will be included.')
        parser.add_argument('--max-depth', dest='max_depth', nargs='?', type=int, default=-1, help='Maximum depth of the workflow execution tree. If 0, only the root workflow execution and its tasks will be included')
        return parser

    def _get_resources(self, parsed_args):
        resource_function = self._get_resources_function()
        errors_only = parsed_args.errors_only or ''
        return resource_function(parsed_args.id, errors_only=errors_only, max_depth=parsed_args.max_depth)

    def take_action(self, parsed_args):
        format_func = self._get_format_function()
        execs_list = self._get_resources(parsed_args)
        if not isinstance(execs_list, list):
            execs_list = [execs_list]
        data = [format_func(r)[1] for r in execs_list]
        return (format_func()[0], data) if data else format_func()