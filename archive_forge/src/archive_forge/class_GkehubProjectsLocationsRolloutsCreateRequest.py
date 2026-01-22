from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsRolloutsCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsRolloutsCreateRequest object.

  Fields:
    parent: Required. The parent resource where this rollout will be created.
      projects/{project}/locations/{location}
    rollout: A Rollout resource to be passed as the request body.
    rolloutId: Required. User provided identifier that is used as part of the
      resource name; must conform to RFC-1034 and additionally restrict to
      lower-cased letters. This comes out roughly to: /^a-z+[a-z0-9]$/
  """
    parent = _messages.StringField(1, required=True)
    rollout = _messages.MessageField('Rollout', 2)
    rolloutId = _messages.StringField(3)