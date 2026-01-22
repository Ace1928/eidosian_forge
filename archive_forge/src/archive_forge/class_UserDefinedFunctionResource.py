from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserDefinedFunctionResource(_messages.Message):
    """A UserDefinedFunctionResource object.

  Fields:
    inlineCode: [Pick one] An inline resource that contains code for a user-
      defined function (UDF). Providing a inline code resource is equivalent
      to providing a URI for a file containing the same code.
    resourceUri: [Pick one] A code resource to load from a Google Cloud
      Storage URI (gs://bucket/path).
  """
    inlineCode = _messages.StringField(1)
    resourceUri = _messages.StringField(2)