from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
Generates Apitools transfer message based on command arguments.