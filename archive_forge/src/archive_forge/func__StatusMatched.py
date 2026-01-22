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
def _StatusMatched(status, repo_status):
    """Checked if the aggregaged repo status matches the given status."""
    if status.lower() == 'all':
        return True
    if status.lower() == 'pending':
        return repo_status.pending > 0
    if status.lower() == 'stalled':
        return repo_status.stalled > 0
    if status.lower() == 'error':
        return repo_status.error > 0
    if status.lower() == 'synced':
        return repo_status.synced == repo_status.total