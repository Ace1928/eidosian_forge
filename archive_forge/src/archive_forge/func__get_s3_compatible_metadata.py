from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _get_s3_compatible_metadata(args, messages):
    """Generates advanced settings for S3-compatible providers."""
    if not (args.source_auth_method or args.source_list_api or args.source_network_protocol or args.source_request_model):
        return None
    s3_compatible_metadata = messages.S3CompatibleMetadata()
    if args.source_auth_method:
        s3_compatible_metadata.authMethod = getattr(messages.S3CompatibleMetadata.AuthMethodValueValuesEnum, 'AUTH_METHOD_' + args.source_auth_method)
    if args.source_list_api:
        s3_compatible_metadata.listApi = getattr(messages.S3CompatibleMetadata.ListApiValueValuesEnum, args.source_list_api)
    if args.source_network_protocol:
        s3_compatible_metadata.protocol = getattr(messages.S3CompatibleMetadata.ProtocolValueValuesEnum, 'NETWORK_PROTOCOL_' + args.source_network_protocol)
    if args.source_request_model:
        s3_compatible_metadata.requestModel = getattr(messages.S3CompatibleMetadata.RequestModelValueValuesEnum, 'REQUEST_MODEL_' + args.source_request_model)
    return s3_compatible_metadata