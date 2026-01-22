from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesCreateRequest object.

  Fields:
    parent: Required. The parent (project and location) where the Scope will
      be created. Specified in the format `projects/*/locations/*`.
    scope: A Scope resource to be passed as the request body.
    scopeId: Required. Client chosen ID for the Scope. `scope_id` must be a
      ????
  """
    parent = _messages.StringField(1, required=True)
    scope = _messages.MessageField('Scope', 2)
    scopeId = _messages.StringField(3)