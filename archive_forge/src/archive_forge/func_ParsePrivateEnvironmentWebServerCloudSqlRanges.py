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
def ParsePrivateEnvironmentWebServerCloudSqlRanges(self, args, image_version, release_track):
    if args.web_server_ipv4_cidr and (not image_versions_util.IsImageVersionStringComposerV1(image_version)):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V2_ERROR_MSG.format(opt='web-server-ipv4-cidr'))
    if args.web_server_ipv4_cidr and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='web-server-ipv4-cidr'))
    if args.cloud_sql_ipv4_cidr and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='cloud-sql-ipv4-cidr'))
    if args.composer_network_ipv4_cidr and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='composer-network-ipv4-cidr'))
    if args.composer_network_ipv4_cidr and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='composer-network-ipv4-cidr'))