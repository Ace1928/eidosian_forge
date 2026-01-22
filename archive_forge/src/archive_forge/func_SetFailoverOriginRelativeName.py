from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def SetFailoverOriginRelativeName(unused_ref, args, request):
    """Parse the provided failover origin to a relative name.

  Relative name includes defaults (or overridden values) for project & location.
  Location defaults to 'global'.

  Args:
    unused_ref: A string representing the operation reference. Unused and may be
      None.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    project = request.parent.split('/')[1]
    request.edgeCacheOrigin.failoverOrigin = resources.REGISTRY.Parse(args.failover_origin, params={'projectsId': args.project or project, 'locationsId': args.location or SetLocationAsGlobal(), 'edgeCacheOriginsId': request.edgeCacheOriginId}, collection='networkservices.projects.locations.edgeCacheOrigins').RelativeName()
    return request