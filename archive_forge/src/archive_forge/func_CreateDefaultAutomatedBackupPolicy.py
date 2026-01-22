from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def CreateDefaultAutomatedBackupPolicy():
    """Constructs AutomatedBackupPolicy message with default values.

  The default values are: retention_period=3d, frequency=1d

  Returns:
    AutomatedBackupPolicy with default policy config.
  """
    return util.GetAdminMessages().AutomatedBackupPolicy(retentionPeriod=ConvertDurationToSeconds('3d'), frequency=ConvertDurationToSeconds('1d'))