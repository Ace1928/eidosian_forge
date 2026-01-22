from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import memcache
def ModifyParams(ref, args, req):
    """Update patch request to include parameters.

  Args:
    ref: The field resource reference.
    args: The parsed arg namespace.
    req: The auto-generated patch request.
  Returns:
    FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest
  """
    if args.IsSpecified('parameters'):
        messages = memcache.Messages(ref.GetCollectionInfo().api_version)
        params = encoding.DictToMessage(args.parameters, messages.MemcacheParameters.ParamsValue)
        parameters = messages.MemcacheParameters(params=params)
        param_req = messages.UpdateParametersRequest(updateMask='params', parameters=parameters)
        req.updateParametersRequest = param_req
    return req