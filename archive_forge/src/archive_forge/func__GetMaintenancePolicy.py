from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import memcache
def _GetMaintenancePolicy(message_module):
    """Returns a maintenance policy of the appropriate version."""
    if hasattr(message_module, 'GoogleCloudMemcacheV1beta2MaintenancePolicy'):
        return message_module.GoogleCloudMemcacheV1beta2MaintenancePolicy()
    elif hasattr(message_module, 'GoogleCloudMemcacheV1MaintenancePolicy'):
        return message_module.GoogleCloudMemcacheV1MaintenancePolicy()
    raise AttributeError('No MaintenancePolicy found for version V1 or V1beta2.')