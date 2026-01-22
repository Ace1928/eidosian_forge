from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsAssessmentsCreateRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsAssessmentsCreateRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1Assessment: A
      GoogleCloudRecaptchaenterpriseV1Assessment resource to be passed as the
      request body.
    parent: Required. The name of the project in which the assessment will be
      created, in the format `projects/{project}`.
  """
    googleCloudRecaptchaenterpriseV1Assessment = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1Assessment', 1)
    parent = _messages.StringField(2, required=True)