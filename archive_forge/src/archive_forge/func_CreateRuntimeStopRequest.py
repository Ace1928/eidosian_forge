from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateRuntimeStopRequest(args, messages):
    runtime = GetRuntimeResource(args).RelativeName()
    stop_request = messages.StopRuntimeRequest()
    return messages.NotebooksProjectsLocationsRuntimesStopRequest(name=runtime, stopRuntimeRequest=stop_request)