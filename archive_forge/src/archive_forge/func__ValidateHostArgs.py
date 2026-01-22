from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _ValidateHostArgs(self, args):
    if not args.IsKnownAndSpecified('host'):
        return True
    pattern = re.compile('[a-zA-Z0-9][-.a-zA-Z0-9]*[a-zA-Z0-9]')
    if not pattern.match(args.host):
        raise calliope_exceptions.InvalidArgumentException('host', 'Hostname and IP can only include letters, numbers, dots, hyphens and valid IP ranges.')