from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateRuntimeCreateRequest(args, messages):
    parent = util.GetParentForRuntime(args)
    runtime = CreateRuntime(args, messages)
    return messages.NotebooksProjectsLocationsRuntimesCreateRequest(parent=parent, runtime=runtime, runtimeId=args.runtime)