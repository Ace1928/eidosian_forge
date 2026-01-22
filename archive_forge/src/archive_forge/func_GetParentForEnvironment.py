from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetParentForEnvironment(args):
    if args.IsSpecified('environment'):
        environment = args.CONCEPTS.environment.Parse()
        return GetLocationResource(environment.locationsId, environment.projectsId).RelativeName()