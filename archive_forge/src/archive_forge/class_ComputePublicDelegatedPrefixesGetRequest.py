from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputePublicDelegatedPrefixesGetRequest(_messages.Message):
    """A ComputePublicDelegatedPrefixesGetRequest object.

  Fields:
    project: Project ID for this request.
    publicDelegatedPrefix: Name of the PublicDelegatedPrefix resource to
      return.
    region: Name of the region of this request.
  """
    project = _messages.StringField(1, required=True)
    publicDelegatedPrefix = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)