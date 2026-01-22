from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsDeleteRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsDeleteRequest object.

  Fields:
    name: Required. The name of the function which should be deleted.
  """
    name = _messages.StringField(1, required=True)