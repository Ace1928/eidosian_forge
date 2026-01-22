from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import urllib.parse as urlparse
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.services.exceptions import GetServicePermissionDeniedException
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
Ensure the given APIs are enabled for the specified project.