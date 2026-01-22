from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceId(_messages.Message):
    """A container to reference an id for any resource type. A `resource` in
  Google Cloud Platform is a generic term for something you (a developer) may
  want to interact with through one of our API's. Some examples are an App
  Engine app, a Compute Engine instance, a Cloud SQL database, and so on.

  Fields:
    id: The type-specific id. This should correspond to the id used in the
      type-specific API's.
    type: The resource type this id is for. At present, the valid types are:
      "organization", "folder", and "project".
  """
    id = _messages.StringField(1)
    type = _messages.StringField(2)