from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_command_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ServiceNetworkingException(core_exceptions.Error):
    """Exception for creation failures involving Service Networking/Peering."""