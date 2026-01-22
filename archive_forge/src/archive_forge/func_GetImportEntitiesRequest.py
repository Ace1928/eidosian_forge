from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.api_lib.datastore import util
def GetImportEntitiesRequest(project, input_url, kinds=None, namespaces=None, labels=None):
    """Returns a request for a Datastore Admin Import.

  Args:
    project: the project id to import, a string.
    input_url: the location of the GCS overall export file, a string.
    kinds: a string list of kinds to import.
    namespaces:  a string list of namespaces to import.
    labels: a string->string map of client labels.
  Returns:
    an ImportRequest message.
  """
    messages = util.GetMessages()
    request_class = messages.GoogleDatastoreAdminV1ImportEntitiesRequest
    entity_filter = _MakeEntityFilter(namespaces, kinds)
    labels_message = request_class.LabelsValue()
    labels_message.additionalProperties = []
    labels = labels or {}
    for key, value in sorted(labels.items()):
        labels_message.additionalProperties.append(request_class.LabelsValue.AdditionalProperty(key=key, value=value))
    import_request = request_class(labels=labels_message, entityFilter=entity_filter, inputUrl=input_url)
    return messages.DatastoreProjectsImportRequest(projectId=project, googleDatastoreAdminV1ImportEntitiesRequest=import_request)