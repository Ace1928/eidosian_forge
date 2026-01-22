from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesExecutionsGetRequest(_messages.Message):
    """A RunNamespacesExecutionsGetRequest object.

  Fields:
    name: Required. The name of the execution to retrieve. Replace {namespace}
      with the project ID or number. It takes the form namespaces/{namespace}.
      For example: namespaces/PROJECT_ID
  """
    name = _messages.StringField(1, required=True)