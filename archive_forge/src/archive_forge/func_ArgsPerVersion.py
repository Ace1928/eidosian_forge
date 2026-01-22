from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
@staticmethod
def ArgsPerVersion(api_version, parser):
    resource_args.AddLocationResourceArg(parser, 'to list registrations for')
    parser.display_info.AddFormat(_FORMAT)
    parser.display_info.AddUriFunc(util.RegistrationsUriFunc(api_version))