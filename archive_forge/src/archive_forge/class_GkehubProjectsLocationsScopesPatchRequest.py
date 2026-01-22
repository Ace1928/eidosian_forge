from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesPatchRequest object.

  Fields:
    name: The resource name for the scope
      `projects/{project}/locations/{location}/scopes/{scope}`
    scope: A Scope resource to be passed as the request body.
    updateMask: Required. The fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    scope = _messages.MessageField('Scope', 2)
    updateMask = _messages.StringField(3)