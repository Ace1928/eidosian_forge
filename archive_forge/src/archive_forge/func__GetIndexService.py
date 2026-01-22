from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def _GetIndexService():
    """Returns the Firestore Index service for interacting with the Firestore Admin service."""
    return api_utils.GetClient().projects_databases_collectionGroups_indexes