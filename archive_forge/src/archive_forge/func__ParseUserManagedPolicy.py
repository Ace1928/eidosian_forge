from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ParseUserManagedPolicy(user_managed_policy):
    """"Reads user managed replication policy file and returns its data.

  Args:
      user_managed_policy (str): The json user managed message

  Returns:
      result (str): "user-managed"
      locations (list): Locations that are part of the user-managed replication
      keys (list): list of kms keys to be used for each replica.
  """
    if 'replicas' not in user_managed_policy or not user_managed_policy['replicas']:
        raise exceptions.BadFileException('Failed to find any replicas in user_managed policy.')
    keys = []
    locations = []
    for replica in user_managed_policy['replicas']:
        if 'location' not in replica:
            raise exceptions.BadFileException('Failed to find a location in all replicas.')
        locations.append(replica['location'])
        if 'customerManagedEncryption' in replica:
            if 'kmsKeyName' in replica['customerManagedEncryption']:
                keys.append(replica['customerManagedEncryption']['kmsKeyName'])
            else:
                raise exceptions.BadFileException('Failed to find a kmsKeyName in customerManagedEncryption for replica at least one replica.')
        if keys and len(keys) != len(locations):
            raise exceptions.BadFileException('Only some replicas have customerManagedEncryption. Please either add the missing field to the remaining replicas or remove it from all replicas.')
    return ('user-managed', locations, keys)