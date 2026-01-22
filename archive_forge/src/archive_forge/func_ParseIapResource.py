from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def ParseIapResource(release_track, args):
    """Parse an IAP resource from the input arguments.

  Args:
    release_track: base.ReleaseTrack, release track of command.
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Raises:
    calliope_exc.InvalidArgumentException: if `--version` was specified with
        resource type 'backend-services'.
    iap_exc.InvalidIapIamResourceError: if an IapIamResource could not be parsed
        from the arguments.

  Returns:
    The specified IapIamResource
  """
    project = properties.VALUES.core.project.GetOrFail()
    if args.resource_type:
        if args.resource_type == APP_ENGINE_RESOURCE_TYPE:
            if args.service:
                raise calliope_exc.InvalidArgumentException('--service', '`--service` cannot be specified for `--resource-type=app-engine`.')
            return iap_api.AppEngineApplication(release_track, project)
        elif args.resource_type == BACKEND_SERVICES_RESOURCE_TYPE:
            if not args.service:
                raise calliope_exc.RequiredArgumentException('--service', '`--service` must be specified for `--resource-type=backend-services`.')
            return iap_api.BackendService(release_track, project, None, args.service)
    raise iap_exc.InvalidIapIamResourceError('Could not parse IAP resource.')