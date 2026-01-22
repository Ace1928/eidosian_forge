from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.backup_restore.poller import BackupPoller
from googlecloudsdk.api_lib.container.backup_restore.poller import RestorePoller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
def _BackupStatusUpdate(result, unused_state):
    del unused_state
    log.Print('Waiting for backup to complete... Backup state: {0}.'.format(result.state))