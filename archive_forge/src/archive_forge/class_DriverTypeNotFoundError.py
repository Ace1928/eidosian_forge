from typing import Dict
from libcloud.dns.providers import Provider as DnsProvider
from libcloud.dns.providers import get_driver as get_dns_driver
from libcloud.backup.providers import Provider as BackupProvider
from libcloud.backup.providers import get_driver as get_backup_driver
from libcloud.compute.providers import Provider as ComputeProvider
from libcloud.compute.providers import get_driver as get_compute_driver
from libcloud.storage.providers import Provider as StorageProvider
from libcloud.storage.providers import get_driver as get_storage_driver
from libcloud.container.providers import Provider as ContainerProvider
from libcloud.container.providers import get_driver as get_container_driver
from libcloud.loadbalancer.providers import Provider as LoadBalancerProvider
from libcloud.loadbalancer.providers import get_driver as get_loadbalancer_driver
class DriverTypeNotFoundError(KeyError):

    def __init__(self, type):
        self.message = "Driver type '%s' not found." % type

    def __repr__(self):
        return self.message