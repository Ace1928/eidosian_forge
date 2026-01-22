import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _snapshot_exists(self, session, backing):
    snapshot = session.invoke_api(vim_util, 'get_object_property', session.vim, backing, 'snapshot')
    if snapshot is None or snapshot.rootSnapshotList is None:
        return False
    return len(snapshot.rootSnapshotList) != 0