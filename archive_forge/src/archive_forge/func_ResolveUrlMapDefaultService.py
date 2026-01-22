from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import operation_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
def ResolveUrlMapDefaultService(args, backend_service_arg, url_map_ref, resources):
    """Parses the backend service that is pointed to by a URL map from args.

  This function handles parsing a regional/global backend service that is
  pointed to by a regional/global URL map.

  Args:
    args: The arguments provided to the url-maps command
    backend_service_arg: The ResourceArgument specification for the
                         backend service argument.
    url_map_ref: The resource reference to the URL MAP. This is returned by
                 parsing the URL map arguments provided.
    resources: ComputeApiHolder resources.

  Returns:
    Backend service reference parsed from args.
  """
    if not compute_scope.IsSpecifiedForFlag(args, 'default_service'):
        if IsRegionalUrlMapRef(url_map_ref):
            args.default_service_region = url_map_ref.region
        else:
            args.global_default_service = args.default_service
    return backend_service_arg.ResolveAsResource(args, resources)