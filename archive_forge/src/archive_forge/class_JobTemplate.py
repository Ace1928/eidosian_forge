from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobTemplate(_messages.Message):
    """Transcoding job template resource.

  Messages:
    LabelsValue: The labels associated with this job template. You can use
      these to organize and group your job templates.

  Fields:
    config: The configuration for this template.
    labels: The labels associated with this job template. You can use these to
      organize and group your job templates.
    name: The resource name of the job template. Format: `projects/{project_nu
      mber}/locations/{location}/jobTemplates/{job_template}`
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels associated with this job template. You can use these to
    organize and group your job templates.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    config = _messages.MessageField('JobConfig', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)