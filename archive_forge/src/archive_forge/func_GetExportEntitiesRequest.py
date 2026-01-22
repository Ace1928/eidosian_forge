from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.api_lib.datastore import util
def GetExportEntitiesRequest(project, output_url_prefix, kinds=None, namespaces=None, labels=None):
    """Returns a request for a Datastore Admin Export.

  Args:
    project: the project id to export, a string.
    output_url_prefix: the output GCS path prefix, a string.
    kinds: a string list of kinds to export.
    namespaces:  a string list of namespaces to export.
    labels: a string->string map of client labels.
  Returns:
    an ExportRequest message.
  """
    messages = util.GetMessages()
    request_class = messages.GoogleDatastoreAdminV1ExportEntitiesRequest
    labels_message = request_class.LabelsValue()
    labels_message.additionalProperties = []
    labels = labels or {}
    for key, value in sorted(labels.items()):
        labels_message.additionalProperties.append(request_class.LabelsValue.AdditionalProperty(key=key, value=value))
    entity_filter = _MakeEntityFilter(namespaces, kinds)
    export_request = request_class(labels=labels_message, entityFilter=entity_filter, outputUrlPrefix=output_url_prefix)
    request = messages.DatastoreProjectsExportRequest(projectId=project, googleDatastoreAdminV1ExportEntitiesRequest=export_request)
    return request