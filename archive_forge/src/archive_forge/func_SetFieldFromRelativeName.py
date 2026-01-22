from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def SetFieldFromRelativeName(api_field):

    def Process(ref, args, request):
        del args
        arg_utils.SetFieldInMessage(request, api_field, ref.RelativeName())
        return request
    return Process