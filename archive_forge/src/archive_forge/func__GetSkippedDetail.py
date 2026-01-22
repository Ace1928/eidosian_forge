from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetSkippedDetail(outcome):
    """Build a string with skippedDetail if present."""
    if outcome.skippedDetail:
        if outcome.skippedDetail.incompatibleDevice:
            return 'Incompatible device/OS combination'
        if outcome.skippedDetail.incompatibleArchitecture:
            return 'App architecture or requested options are incompatible with this device'
        if outcome.skippedDetail.incompatibleAppVersion:
            return 'App does not support the OS version'
    return 'Unknown reason'