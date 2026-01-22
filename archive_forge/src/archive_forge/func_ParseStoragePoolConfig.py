from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseStoragePoolConfig(self, name, service_level, network, kms_config, active_directory, enable_ldap, capacity, description, allow_auto_tiering, labels):
    """Parses the command line arguments for Create Storage Pool into a config.

    Args:
      name: the name of the Storage Pool
      service_level: the service level of the Storage Pool
      network: the network of the Storage Pool
      kms_config: the KMS Config of the Storage Pool
      active_directory: the Active Directory of the Storage Pool
      enable_ldap: Bool on whether to enable LDAP on Storage Pool
      capacity: the storage capacity of the Storage Pool
      description: the description of the Storage Pool
      allow_auto_tiering: Bool on whether Storage Pool supports auto tiering
      labels: the parsed labels value

    Returns:
      The configuration that will be used as the request body for creating a
      Cloud NetApp Storage Pool.
    """
    storage_pool = self.messages.StoragePool()
    storage_pool.name = name
    storage_pool.serviceLevel = service_level
    storage_pool.network = network.get('name')
    if 'psa-range' in network:
        storage_pool.psaRange = network.get('psa-range')
    storage_pool.kmsConfig = kms_config
    storage_pool.activeDirectory = active_directory
    storage_pool.ldapEnabled = enable_ldap
    storage_pool.capacityGib = capacity
    storage_pool.description = description
    if allow_auto_tiering is not None:
        storage_pool.allowAutoTiering = allow_auto_tiering
    storage_pool.labels = labels
    return storage_pool