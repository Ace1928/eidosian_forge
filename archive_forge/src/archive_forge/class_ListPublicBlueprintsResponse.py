from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPublicBlueprintsResponse(_messages.Message):
    """Response object for `ListPublicBlueprints`.

  Fields:
    nextPageToken: Output only. A token identifying a page of results the
      server should return.
    publicBlueprints: The list of public blueprints to return.
  """
    nextPageToken = _messages.StringField(1)
    publicBlueprints = _messages.MessageField('PublicBlueprint', 2, repeated=True)