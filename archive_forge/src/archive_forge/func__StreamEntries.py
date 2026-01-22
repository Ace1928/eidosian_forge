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
def _StreamEntries(get_now, output_warning, output_error, output_debug, tail_stub):
    """Streams entries back from the Logging API.

  Args:
    get_now: A callable that returns the current time.
    output_warning: A callable that outputs the argument as a warning.
    output_error: A callable that outputs the argument as an error.
    output_debug: A callable that outputs the argument as debug info.
    tail_stub: The `BidiRpc` stub to use.

  Yields:
    Entries included in the tail session.
  """
    tail_stub.open()
    suppression_info_accumulator = _SuppressionInfoAccumulator(get_now, output_warning, output_error)
    error = None
    while tail_stub.is_active:
        try:
            response = tail_stub.recv()
        except grpc.RpcError as e:
            error = e
            break
        suppression_info_accumulator.Add(response.suppression_info)
        for entry in response.entries:
            yield entry
    if error:
        _HandleGrpcRendezvous(error, output_debug, output_warning)
    suppression_info_accumulator.Finish()
    tail_stub.close()