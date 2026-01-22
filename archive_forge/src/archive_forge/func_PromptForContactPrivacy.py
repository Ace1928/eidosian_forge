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
def PromptForContactPrivacy(api_version, choices, current_privacy=None):
    """Asks a user for Contacts Privacy.

  Args:
    api_version: Cloud Domains API version to call.
    choices: List of privacy choices.
    current_privacy: Current privacy. Should be nonempty in update calls.

  Returns:
    Privacy enum or None if the user cancelled.
  """
    if not choices:
        raise exceptions.Error('Could not find supported contact privacy.')
    domains_messages = registrations.GetMessagesModule(api_version)
    choices.sort(key=flags.PrivacyChoiceStrength, reverse=True)
    if current_privacy:
        if len(choices) == 1:
            log.status.Print('Your current contact privacy is {}. It cannot be changed.'.format(current_privacy))
            return None
        else:
            update = console_io.PromptContinue('Your current contact privacy is {}.'.format(current_privacy), 'Do you want to change it', default=False)
            if not update:
                return None
        current_choice = 0
        for ix, privacy in enumerate(choices):
            if privacy == flags.ContactPrivacyEnumMapper(domains_messages).GetChoiceForEnum(current_privacy):
                current_choice = ix
    else:
        current_choice = 0
    if len(choices) == 1:
        ack = console_io.PromptContinue('The only supported contact privacy is {}.'.format(choices[0]), default=True)
        if not ack:
            return None
        return ParseContactPrivacy(api_version, choices[0])
    else:
        index = console_io.PromptChoice(options=choices, default=current_choice, message='Specify contact privacy')
        return ParseContactPrivacy(api_version, choices[index])