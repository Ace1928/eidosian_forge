from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetServiceProjectRef(args):
    """Returns a service project reference."""
    service_project_ref = args.CONCEPTS.service_project.Parse()
    if not service_project_ref.Name():
        raise exceptions.InvalidArgumentException('service project', 'service project id must be non-empty.')
    return service_project_ref