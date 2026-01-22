from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
import six
def ShouldDoUpdateCheck(self):
    """Checks if it is time to do an update check.

    Returns:
      True, if enough time has elapsed and we should perform another update
      check.  False otherwise.
    """
    return self.SecondsSinceLastUpdateCheck() >= UpdateCheckData.UPDATE_CHECK_FREQUENCY_IN_SECONDS