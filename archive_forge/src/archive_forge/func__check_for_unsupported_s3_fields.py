from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
def _check_for_unsupported_s3_fields(user_request_args):
    """Raises error or logs warning if unsupported S3 field present."""
    user_resource_args = getattr(user_request_args, 'resource_args', None)
    if user_resource_args and (not getattr(user_resource_args, 'retention_period_to_be_locked', None)):
        user_resource_args.retention_period_to_be_locked = None
    error_fields_present = _extract_unsupported_features_from_user_args(user_request_args, S3_REQUEST_ERROR_FIELDS) + _extract_unsupported_features_from_user_args(user_resource_args, S3_RESOURCE_ERROR_FIELDS)
    if error_fields_present:
        raise errors.Error('Features disallowed for S3: {}'.format(', '.join(error_fields_present)))
    warning_fields_present = _extract_unsupported_features_from_user_args(user_resource_args, S3_RESOURCE_WARNING_FIELDS)
    if warning_fields_present:
        log.warning('Some features do not have S3 support: {}'.format(', '.join(warning_fields_present)))