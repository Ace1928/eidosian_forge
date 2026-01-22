from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def _GetContactNoticeEnum(domains_messages):
    """Get ContactNoticeEnum from api messages."""
    return domains_messages.ConfigureContactSettingsRequest.ContactNoticesValueListEntryValuesEnum