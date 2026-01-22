from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def SetFieldFromArg(api_field, arg_name):

    def Process(unused_ref, args, req):
        arg_utils.SetFieldInMessage(req, api_field, arg_utils.GetFromNamespace(args, arg_name))
        return req
    return Process