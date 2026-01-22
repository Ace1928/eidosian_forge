from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsTemplatesGetRequest(_messages.Message):
    """A DataflowProjectsTemplatesGetRequest object.

  Enums:
    ViewValueValuesEnum: The view to retrieve. Defaults to METADATA_ONLY.

  Fields:
    gcsPath: Required. A Cloud Storage path to the template from which to
      create the job. Must be valid Cloud Storage URL, beginning with 'gs://'.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
    view: The view to retrieve. Defaults to METADATA_ONLY.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view to retrieve. Defaults to METADATA_ONLY.

    Values:
      METADATA_ONLY: Template view that retrieves only the metadata associated
        with the template.
    """
        METADATA_ONLY = 0
    gcsPath = _messages.StringField(1)
    location = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)