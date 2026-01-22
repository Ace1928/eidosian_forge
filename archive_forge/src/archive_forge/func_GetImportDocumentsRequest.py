from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def GetImportDocumentsRequest(database, input_uri_prefix, namespace_ids=None, collection_ids=None):
    """Returns a request for a Firestore Admin Import.

  Args:
    database: the database id to import, a string.
    input_uri_prefix: the location of the GCS export files, a string.
    namespace_ids: a string list of namespace ids to import.
    collection_ids: a string list of collection ids to import.

  Returns:
    an ImportDocumentsRequest message.
  """
    messages = api_utils.GetMessages()
    request_class = messages.GoogleFirestoreAdminV1ImportDocumentsRequest
    kwargs = {'inputUriPrefix': input_uri_prefix}
    if collection_ids:
        kwargs['collectionIds'] = collection_ids
    if namespace_ids:
        kwargs['namespaceIds'] = namespace_ids
    import_request = request_class(**kwargs)
    return messages.FirestoreProjectsDatabasesImportDocumentsRequest(name=database, googleFirestoreAdminV1ImportDocumentsRequest=import_request)