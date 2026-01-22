from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetErrorMessages(errors):
    """return the errorMessage list from a list of ConfigSync errors."""
    return_errors = []
    for err in errors:
        return_errors.append(err['errorMessage'])
    return return_errors