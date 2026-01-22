from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def ParseIpAliasConfigOptions(self, args, image_version):
    """Parses the options for VPC-native configuration."""
    if args.enable_ip_alias and (not image_versions_util.IsImageVersionStringComposerV1(image_version)):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='enable-ip-alias'))
    if args.cluster_ipv4_cidr and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='cluster-ipv4-cidr'))
    if args.cluster_secondary_range_name and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='cluster-secondary-range-name'))
    if args.services_ipv4_cidr and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='services-ipv4-cidr'))
    if args.services_secondary_range_name and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='services-secondary-range-name'))
    if self._support_max_pods_per_node and args.max_pods_per_node and (not image_versions_util.IsImageVersionStringComposerV1(image_version)):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='max-pods-per-node'))
    if self._support_max_pods_per_node and args.max_pods_per_node and (not args.enable_ip_alias):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='max-pods-per-node'))
    if args.enable_ip_masq_agent and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='enable-ip-masq-agent'))