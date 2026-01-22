from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.interconnects.attachments import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.interconnects.attachments import flags as attachment_flags
from googlecloudsdk.command_lib.compute.routers import flags as router_flags
from googlecloudsdk.core import log
def PrintPairingKeyEpilog(pairing_key):
    """Prints the pairing key help text upon command completion."""
    message = '      Please use the pairing key to provision the attachment with your partner:\n      {0}\n      '.format(pairing_key)
    log.status.Print(message)