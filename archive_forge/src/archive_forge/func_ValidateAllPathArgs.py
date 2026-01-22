from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def ValidateAllPathArgs(args):
    from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
    if args.IsSpecified('all'):
        if args.IsSpecified('path') and (not os.path.isdir(args.path)):
            raise declarative_client_base.ClientException('Error executing export: "{}" must be a directory when --all is specified.'.format(args.path))