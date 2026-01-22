from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kmsinventory import inventory
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import resource_args
class GetProtectedResourcesSummary(base.Command):
    """Gets the protected resources summary."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddKmsKeyResourceArgForKMS(parser, True, '--keyname')

    def Run(self, args):
        keyname = args.keyname
        return inventory.GetProtectedResourcesSummary(keyname)