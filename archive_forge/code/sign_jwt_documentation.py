from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Sign a JWT with a managed service account key.

  This command signs a JWT using a system-managed service account key.

  If the service account does not exist, this command returns a
  `PERMISSION_DENIED` error.
  