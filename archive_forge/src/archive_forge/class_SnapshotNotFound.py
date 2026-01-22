import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class SnapshotNotFound(EntityNotFound):
    msg_fmt = _('The Snapshot (%(snapshot)s) for Stack (%(stack)s) could not be found.')