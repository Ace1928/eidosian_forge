from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util as projects_api_util
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding as encoder
from googlecloudsdk.core.util import retry
import six
def GetStateMessage(messages):
    """Returns corresponding GoogleCloudFunctionsV2(alpha|beta|ga)stateMessage."""
    if messages is apis.GetMessagesModule(_API_NAME, _V2_ALPHA):
        return messages.GoogleCloudFunctionsV2alphaStateMessage
    elif messages is apis.GetMessagesModule(_API_NAME, _V2_BETA):
        return messages.GoogleCloudFunctionsV2betaStateMessage
    else:
        return messages.GoogleCloudFunctionsV2StateMessage