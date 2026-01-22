import sys
from os_win._i18n import _
class ClusterGroupMigrationTimeOut(ClusterGroupMigrationFailed):
    msg_fmt = _("Cluster group '%(group_name)s' migration timed out after %(time_elapsed)0.3fs. ")