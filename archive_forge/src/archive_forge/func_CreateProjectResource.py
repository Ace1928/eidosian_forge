from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def CreateProjectResource(args):
    return resources.REGISTRY.Create('sourcerepo.projects', projectsId=args.project or properties.VALUES.core.project.GetOrFail())