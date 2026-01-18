from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def update_replica_count(current, value):
    """Configures a replica count for the current deployment configuration."""
    if value is None:
        current.replicaCount = None
    else:
        current.replicaCount = int(value)
    return current