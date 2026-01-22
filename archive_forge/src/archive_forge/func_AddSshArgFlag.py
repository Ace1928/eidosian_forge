from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
def AddSshArgFlag(parser):
    parser.add_argument('ssh_args', nargs=argparse.REMAINDER, help='          Flags and positionals passed to the underlying ssh implementation.\n          ', example='        $ {command} -- -vvv\n      ')