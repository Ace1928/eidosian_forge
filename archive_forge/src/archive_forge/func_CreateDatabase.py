from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def CreateDatabase(project, location, database, database_type, delete_protection_state, pitr_state, cmek_config):
    """Performs a Firestore Admin v1 Database Creation.

  Args:
    project: the project id to create, a string.
    location: the database location to create, a string.
    database: the database id to create, a string.
    database_type: the database type, an Enum.
    delete_protection_state: the value for deleteProtectionState, an Enum.
    pitr_state: the value for PitrState, an Enum.
    cmek_config: the CMEK config used to encrypt the database, an object

  Returns:
    an Operation.
  """
    messages = api_utils.GetMessages()
    return _GetDatabaseService().Create(messages.FirestoreProjectsDatabasesCreateRequest(parent='projects/{}'.format(project), databaseId=database, googleFirestoreAdminV1Database=messages.GoogleFirestoreAdminV1Database(type=database_type, locationId=location, deleteProtectionState=delete_protection_state, pointInTimeRecoveryEnablement=pitr_state, cmekConfig=cmek_config)))