import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
class MyCommand(c_cmd.Command):

    def get_parser(self, prog_name):
        parser = super(MyCommand, self).get_parser(prog_name)
        parser.add_argument('--end')
        return parser

    def take_action(self, parsed_args):
        assert parsed_args.end == '123'