from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetFullSourceName(args, version):
    """Returns relative resource name for a source from --source argument.

  Args:
    args: Argument namespace.
    version: Api version.

  Returns:
    Relative resource name
    if args.source is not provided an exception will be raised
    if no location is specified in argument: sources/{source_id}
    if a location is specified: sources/{source_id}/locations/{location_id}
  """
    resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9-]+')
    region_resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9-]+/locations/[a-zA-Z0-9-]+$')
    id_pattern = re.compile('[0-9-]+')
    if not args.source:
        raise errors.InvalidSCCInputError('Finding source must be provided in --source flag or full resource name.')
    if region_resource_pattern.match(args.source):
        return args.source
    location = scc_util.ValidateAndGetLocation(args, version)
    if resource_pattern.match(args.source):
        source = args.source
        if version == 'v2':
            return f'{source}/locations/{location}'
        return source
    if id_pattern.match(args.source):
        source = f'{scc_util.GetParentFromPositionalArguments(args)}/sources/{args.source}'
        if version == 'v2':
            return f'{source}/locations/{location}'
        return source
    raise errors.InvalidSCCInputError('The source must either be the full resource name or the numeric source ID.')