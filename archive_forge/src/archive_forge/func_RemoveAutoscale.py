from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def RemoveAutoscale(ref, args, request):
    del ref
    if args.IsSpecified('clear_autoscale'):
        request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest.workerPool.autoscale = None
        req = request.googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest
        AddFieldToMask('autoscale.min_size', req)
        AddFieldToMask('autoscale.max_size', req)
    return request