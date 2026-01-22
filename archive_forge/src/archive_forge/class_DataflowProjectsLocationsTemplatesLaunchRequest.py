from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsTemplatesLaunchRequest(_messages.Message):
    """A DataflowProjectsLocationsTemplatesLaunchRequest object.

  Fields:
    dynamicTemplate_gcsPath: Path to the dynamic template specification file
      on Cloud Storage. The file must be a JSON serialized
      `DynamicTemplateFileSpec` object.
    dynamicTemplate_stagingLocation: Cloud Storage path for staging
      dependencies. Must be a valid Cloud Storage URL, beginning with `gs://`.
    gcsPath: A Cloud Storage path to the template to use to create the job.
      Must be valid Cloud Storage URL, beginning with `gs://`.
    launchTemplateParameters: A LaunchTemplateParameters resource to be passed
      as the request body.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
    validateOnly: If true, the request is validated but not actually executed.
      Defaults to false.
  """
    dynamicTemplate_gcsPath = _messages.StringField(1)
    dynamicTemplate_stagingLocation = _messages.StringField(2)
    gcsPath = _messages.StringField(3)
    launchTemplateParameters = _messages.MessageField('LaunchTemplateParameters', 4)
    location = _messages.StringField(5, required=True)
    projectId = _messages.StringField(6, required=True)
    validateOnly = _messages.BooleanField(7)