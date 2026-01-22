from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def AddOtherPropertiesToRequest(unused_instance_ref, args, request):
    intent = encoding.MessageToDict(request.googleCloudDialogflowV2Intent)
    if args.IsSpecified('other_properties'):
        intent.update(args.other_properties)
    request.googleCloudDialogflowV2Intent = encoding.DictToMessage(intent, type(request.googleCloudDialogflowV2Intent))
    return request