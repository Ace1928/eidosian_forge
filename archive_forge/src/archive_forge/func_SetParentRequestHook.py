from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def SetParentRequestHook(ref, args, request):
    """Declarative request hook to add relative parent to issued requests."""
    del args
    request.parent = ref.Parent().RelativeName()
    return request