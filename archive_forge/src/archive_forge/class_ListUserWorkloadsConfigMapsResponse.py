from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUserWorkloadsConfigMapsResponse(_messages.Message):
    """The user workloads ConfigMaps for a given environment.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    userWorkloadsConfigMaps: The list of ConfigMaps returned by a
      ListUserWorkloadsConfigMapsRequest.
  """
    nextPageToken = _messages.StringField(1)
    userWorkloadsConfigMaps = _messages.MessageField('UserWorkloadsConfigMap', 2, repeated=True)