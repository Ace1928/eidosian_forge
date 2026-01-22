from keystoneauth1 import adapter
from keystoneauth1 import session
from keystoneclient import client as ks_client
import manilaclient
from manilaclient.common import constants
from manilaclient.common import httpclient
from manilaclient import exceptions
from manilaclient.v2 import availability_zones
from manilaclient.v2 import limits
from manilaclient.v2 import messages
from manilaclient.v2 import quota_classes
from manilaclient.v2 import quotas
from manilaclient.v2 import resource_locks
from manilaclient.v2 import scheduler_stats
from manilaclient.v2 import security_services
from manilaclient.v2 import services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_backups
from manilaclient.v2 import share_export_locations
from manilaclient.v2 import share_group_snapshots
from manilaclient.v2 import share_group_type_access
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instance_export_locations
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_replica_export_locations
from manilaclient.v2 import share_replicas
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshot_export_locations
from manilaclient.v2 import share_snapshot_instance_export_locations
from manilaclient.v2 import share_snapshot_instances
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_transfers
from manilaclient.v2 import share_type_access
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
def _load_extensions(self, extensions):
    if not extensions:
        return
    for extension in extensions:
        if extension.manager_class:
            setattr(self, extension.name, extension.manager_class(self))