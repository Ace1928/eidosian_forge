from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddPromptForPassword(parser):
    parser.add_argument('--prompt-for-password', action='store_true', help="Prompt for the Cloud SQL user's password with character echo disabled. The password is all typed characters up to but not including the RETURN or ENTER key.")