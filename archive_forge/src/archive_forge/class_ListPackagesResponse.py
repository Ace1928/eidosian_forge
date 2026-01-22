from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPackagesResponse(_messages.Message):
    """The response from listing packages.

  Fields:
    nextPageToken: The token to retrieve the next page of packages, or empty
      if there are no more packages to return.
    packages: The packages returned.
  """
    nextPageToken = _messages.StringField(1)
    packages = _messages.MessageField('Package', 2, repeated=True)