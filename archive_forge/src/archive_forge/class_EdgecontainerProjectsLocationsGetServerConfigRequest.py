from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsGetServerConfigRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsGetServerConfigRequest object.

  Fields:
    name: Required. The name (project and location) of the server config to
      get, specified in the format `projects/*/locations/*`.
  """
    name = _messages.StringField(1, required=True)