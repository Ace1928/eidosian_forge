from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.ai.persistent_resources import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai.persistent_resources import flags
from googlecloudsdk.command_lib.ai.persistent_resources import validation
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Reboot an active Persistent Resource.

  ## EXAMPLES

  To reboot a persistent resource ``123'' under project ``example'' in region
  ``us-central1'', run:

    $ {command} 123 --project=example --region=us-central1
  