from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesJobsReplaceJobRequest(_messages.Message):
    """A RunNamespacesJobsReplaceJobRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    name: Required. The name of the job being replaced. Replace {namespace}
      with the project ID or number. It takes the form namespaces/{namespace}.
      For example: namespaces/PROJECT_ID
  """
    job = _messages.MessageField('Job', 1)
    name = _messages.StringField(2, required=True)