from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import util
from googlecloudsdk.api_lib.firestore import api_utils as firestore_utils
from googlecloudsdk.api_lib.firestore import indexes as firestore_indexes
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def ListDatastoreIndexesViaFirestoreApi(project_id, database_id):
    """Lists all datastore indexes under a database with Firestore Admin API.

  Args:
    project_id: A str to represent the project id.
    database_id: A str to represent the database id.

  Returns:
    List[index]: A list of datastore_index.Index that contains index definition.
  """
    response = firestore_indexes.ListIndexes(project_id, database_id)
    return {FirestoreApiMessageToIndexDefinition(index) for index in response.indexes if index.apiScope == DATASTORE_API_SCOPE}