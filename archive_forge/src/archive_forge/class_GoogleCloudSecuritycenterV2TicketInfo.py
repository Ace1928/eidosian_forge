from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2TicketInfo(_messages.Message):
    """Information about the ticket, if any, that is being used to track the
  resolution of the issue that is identified by this finding.

  Fields:
    assignee: The assignee of the ticket in the ticket system.
    description: The description of the ticket in the ticket system.
    id: The identifier of the ticket in the ticket system.
    status: The latest status of the ticket, as reported by the ticket system.
    updateTime: The time when the ticket was last updated, as reported by the
      ticket system.
    uri: The link to the ticket in the ticket system.
  """
    assignee = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.StringField(3)
    status = _messages.StringField(4)
    updateTime = _messages.StringField(5)
    uri = _messages.StringField(6)