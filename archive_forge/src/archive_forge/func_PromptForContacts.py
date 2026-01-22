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
def PromptForContacts(api_version, current_contacts=None):
    """Interactively prompts for Whois Contact information."""
    domains_messages = registrations.GetMessagesModule(api_version)
    create_call = current_contacts is None
    if not console_io.PromptContinue('Contact data not provided using the --contact-data-from-file flag.', prompt_string='Do you want to enter it interactively', default=create_call):
        return None
    if create_call:
        contact = _PromptForSingleContact(domains_messages)
        return domains_messages.ContactSettings(registrantContact=contact, adminContact=contact, technicalContact=contact)
    choices = ['all the contacts to the same value', 'registrant contact', 'admin contact', 'technical contact']
    index = console_io.PromptChoice(options=choices, cancel_option=True, default=0, message='Which contact do you want to change?')
    if index == 0:
        contact = _PromptForSingleContact(domains_messages, current_contacts.registrantContact)
        return domains_messages.ContactSettings(registrantContact=contact, adminContact=contact, technicalContact=contact)
    if index == 1:
        contact = _PromptForSingleContact(domains_messages, current_contacts.registrantContact)
        return domains_messages.ContactSettings(registrantContact=contact)
    if index == 2:
        contact = _PromptForSingleContact(domains_messages, current_contacts.adminContact)
        return domains_messages.ContactSettings(adminContact=contact)
    if index == 3:
        contact = _PromptForSingleContact(domains_messages, current_contacts.technicalContact)
        return domains_messages.ContactSettings(technicalContact=contact)
    return None