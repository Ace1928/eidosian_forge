from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def PrintSQSPAck(self):
    """Prints FYI acknowledgement about the Google Domains - Squarespace deal.
    """
    ack_message = "Domain name registration services are provided by Squarespace (https://domains.squarespace.com),\npursuant to the Squarespace Terms of Service (https://www.squarespace.com/terms-of-service)\nand Squarespace Domain Registration Agreement (https://www.squarespace.com/domain-registration-agreement),\nwhich Google resells pursuant to an agreement with Squarespace.\nInitially, Google will manage your domain(s) on Squarespace's behalf. Once your domain is transitioned to Squarespace,\nGoogle will share your name, contact information, and other domain-related information with Squarespace.\nYou can review Squarespace's Privacy Policy (https://www.squarespace.com/privacy) for details on how they process your information.\nGoogle's Privacy Policy (https://policies.google.com/privacy) describes how Google handles this information as a reseller.\nBy choosing to continue, you (1) acknowledge receipt of Google's Privacy Policy and direct us to share this information\nwith Squarespace; and (2) agree to the Squarespace Terms of Service (https://www.squarespace.com/terms-of-service) and\nSquarespace Domain Registration Agreement (https://www.squarespace.com/domain-registration-agreement), and\nacknowledge receipt of Squarespace's Privacy Policy (https://www.squarespace.com/privacy).\n"
    log.status.Print(ack_message)