from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ParseReplicationDict(replication_policy):
    """Reads replication policy dictionary and returns its data."""
    if 'userManaged' in replication_policy:
        return _ParseUserManagedPolicy(replication_policy['userManaged'])
    if 'automatic' in replication_policy:
        return _ParseAutomaticPolicy(replication_policy['automatic'])
    raise exceptions.BadFileException('Expected to find either "userManaged" or "automatic" in replication, but found neither.')