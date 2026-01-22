from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseOperationResourceArg(args):
    return resources.REGISTRY.ParseRelativeName(args.CONCEPTS.operation_id.Parse().RelativeName(), collection='gkemulticloud.projects.locations.operations')