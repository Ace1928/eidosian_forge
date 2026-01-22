from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.protorpclite import messages as _messages
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
def ParsePublicContactsAck(api_version, notices):
    """Parses Contact Notices. Returns public_contact_ack enum or None."""
    domains_messages = registrations.GetMessagesModule(api_version)
    if notices is None:
        return False
    for notice in notices:
        enum = flags.ContactNoticeEnumMapper(domains_messages).GetEnumForChoice(notice)
        if enum == domains_messages.ConfigureContactSettingsRequest.ContactNoticesValueListEntryValuesEnum.PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT:
            return enum
    return None