from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.edge_cloud.container import resource_args as gdce_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetGdceClusterResourcePresentationSpec():
    return presentation_specs.ResourcePresentationSpec(name='--gdce-cluster', concept_spec=gdce_resource_args.GetClusterResourceSpec(), group_help='The GDCE cluster on which to create the service instance.', required=True, prefixes=True, flag_name_overrides={'location': ''})