from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMigratingVmsResponse(_messages.Message):
    """Response message for 'ListMigratingVms' request.

  Fields:
    migratingVms: Output only. The list of Migrating VMs response.
    nextPageToken: Output only. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    unreachable: Output only. Locations that could not be reached.
  """
    migratingVms = _messages.MessageField('MigratingVm', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)