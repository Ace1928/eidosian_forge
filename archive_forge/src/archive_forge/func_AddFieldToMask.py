from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddFieldToMask(field, request):
    if request.updateMask:
        if field not in request.updateMask:
            request.updateMask = request.updateMask + ',' + field
    else:
        request.updateMask = field
    return request