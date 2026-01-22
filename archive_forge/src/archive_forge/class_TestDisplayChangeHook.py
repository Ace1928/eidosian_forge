from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
class TestDisplayChangeHook(hooks.CommandHook):
    _before_called = False
    _after_called = False

    def get_parser(self, parser):
        parser.add_argument('--added-by-hook')
        return parser

    def get_epilog(self):
        return 'hook epilog'

    def before(self, parsed_args):
        self._before_called = True
        parsed_args.added_by_hook = 'othervalue'
        parsed_args.added_by_before = True
        return parsed_args

    def after(self, parsed_args, return_code):
        self._after_called = True
        return (('Name',), ('othervalue',))