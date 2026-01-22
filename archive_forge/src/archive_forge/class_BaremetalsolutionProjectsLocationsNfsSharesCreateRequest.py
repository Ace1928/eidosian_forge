from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsNfsSharesCreateRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsNfsSharesCreateRequest object.

  Fields:
    nfsShare: A NfsShare resource to be passed as the request body.
    parent: Required. The parent project and location.
  """
    nfsShare = _messages.MessageField('NfsShare', 1)
    parent = _messages.StringField(2, required=True)