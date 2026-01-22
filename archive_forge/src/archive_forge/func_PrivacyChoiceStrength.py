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
def PrivacyChoiceStrength(privacy):
    """Returns privacy strength (stronger privacy means higher returned value)."""
    if privacy == 'public-contact-data':
        return 0
    if privacy == 'redacted-contact-data':
        return 1
    if privacy == 'private-contact-data':
        return 2