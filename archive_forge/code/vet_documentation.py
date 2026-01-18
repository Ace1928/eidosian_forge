from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
Validate that a terraform plan complies with policies.