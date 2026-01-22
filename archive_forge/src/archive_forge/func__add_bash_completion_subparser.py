import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def _add_bash_completion_subparser(self, subparsers):
    subparser = subparsers.add_parser('bash_completion', add_help=False, formatter_class=OpenStackHelpFormatter)
    self.subcommands['bash_completion'] = subparser
    subparser.set_defaults(func=self.do_bash_completion)