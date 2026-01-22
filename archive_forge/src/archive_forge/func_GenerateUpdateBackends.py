from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.metastore import validators as validator
import six
def GenerateUpdateBackends(job_ref, args, update_federation_req):
    """Construct the long name for backends and updateMask for update requests of Dataproc Metastore federations.

  Args:
    job_ref: A resource ref to the parsed Federation resource.
    args: The parsed args namespace from CLI.
    update_federation_req: Update federation request for the API call.

  Returns:
    Modified request for the API call.
  """
    args_set = set(args.GetSpecifiedArgNames())
    if '--remove-backends' in args_set and '--update-backends' not in args_set:
        update_federation_req.federation.backendMetastores = {}
    if '--update-backends' in args_set:
        validator.ParseBackendsIntoRequest(job_ref, update_federation_req)
    update_federation_req.updateMask = _GenerateUpdateMask(args)
    return update_federation_req