from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddConfigType(parser: parser_arguments.ArgumentInterceptor):
    """Adds flags to specify version config type.

  Args:
    parser: The argparse parser to add the flag to.
  """
    config_type_group = parser.add_group('Use cases for querying versions.', mutex=True, required=False)
    create_config = config_type_group.add_group('Create an Anthos on VMware user cluster use case.')
    upgrade_config = config_type_group.add_group('Upgrade an Anthos on VMware user cluster use case.')
    arg_parser = concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('--admin-cluster-membership', flags.GetAdminClusterMembershipResourceSpec(), 'Membership of the admin cluster to query versions for create. Membership can be the membership ID or the full resource name.', flag_name_overrides={'project': '--admin-cluster-membership-project', 'location': '--admin-cluster-membership-location'}, required=False, group=create_config), presentation_specs.ResourcePresentationSpec('--cluster', GetClusterResourceSpec(), 'Cluster to query versions for upgrade.', required=False, flag_name_overrides={'location': ''}, group=upgrade_config)], command_level_fallthroughs={'--cluster.location': ['--location']})
    arg_parser.AddToParser(parser)
    parser.set_defaults(admin_cluster_membership_location='global')