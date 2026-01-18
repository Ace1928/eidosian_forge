from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.auth import flags
from googlecloudsdk.command_lib.config import config_helper
from googlecloudsdk.core import config
from googlecloudsdk.core.credentials import store as c_store
from oauth2client import client
Run the print_identity_token command.