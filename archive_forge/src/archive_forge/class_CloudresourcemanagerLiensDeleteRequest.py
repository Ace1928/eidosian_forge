from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerLiensDeleteRequest(_messages.Message):
    """A CloudresourcemanagerLiensDeleteRequest object.

  Fields:
    liensId: Part of `name`. Required. The name/identifier of the Lien to
      delete.
  """
    liensId = _messages.StringField(1, required=True)