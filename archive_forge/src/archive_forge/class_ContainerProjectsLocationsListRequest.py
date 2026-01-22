from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsListRequest(_messages.Message):
    """A ContainerProjectsLocationsListRequest object.

  Fields:
    parent: Required. Contains the name of the resource requested. Specified
      in the format `projects/*`.
  """
    parent = _messages.StringField(1, required=True)