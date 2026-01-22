from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseRestoreParameters(self, volume, snapshot, backup):
    """Parses Restore Parameters for Volume into a config."""
    restore_parameters = self.messages.RestoreParameters()
    if snapshot:
        restore_parameters.sourceSnapshot = snapshot
    if backup:
        restore_parameters.sourceBackup = backup
    volume.restoreParameters = restore_parameters