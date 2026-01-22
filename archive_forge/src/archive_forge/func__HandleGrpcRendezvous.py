from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import google.appengine.logging.v1.request_log_pb2
import google.cloud.appengine_v1.proto.audit_data_pb2
import google.cloud.appengine_v1alpha.proto.audit_data_pb2
import google.cloud.appengine_v1beta.proto.audit_data_pb2
import google.cloud.bigquery_logging_v1.proto.audit_data_pb2
import google.cloud.cloud_audit.proto.audit_log_pb2
import google.cloud.iam_admin_v1.proto.audit_data_pb2
import google.iam.v1.logging.audit_data_pb2
import google.type.money_pb2
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import gapic_util
from googlecloudsdk.core import log
import grpc
def _HandleGrpcRendezvous(rendezvous, output_debug, output_warning):
    """Handles _MultiThreadedRendezvous errors."""
    error_messages_by_code = {grpc.StatusCode.INVALID_ARGUMENT: 'Invalid argument.', grpc.StatusCode.RESOURCE_EXHAUSTED: 'There are too many tail sessions open.', grpc.StatusCode.INTERNAL: 'Internal error.', grpc.StatusCode.PERMISSION_DENIED: 'Access is denied or has changed for resource.', grpc.StatusCode.OUT_OF_RANGE: 'The maximum duration for tail has been met. The command may be repeated to continue.'}
    if rendezvous.code() == grpc.StatusCode.CANCELLED:
        return
    output_debug(rendezvous)
    output_warning('{} ({})'.format(error_messages_by_code.get(rendezvous.code(), 'Unknown error encountered.'), rendezvous.details()))