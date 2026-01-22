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
def TailLogs(self, resource_names, logs_filter, buffer_window_seconds=None, output_warning=log.err.Print, output_error=log.error, output_debug=log.debug, get_now=datetime.datetime.now):
    """Tails log entries from the Cloud Logging API.

    Args:
      resource_names: The resource names to tail.
      logs_filter: The Cloud Logging filter identifying entries to include in
        the session.
      buffer_window_seconds: The amount of time that Cloud Logging should buffer
        entries to get correct ordering, or None if the backend should use its
        default.
      output_warning: A callable that outputs the argument as a warning.
      output_error: A callable that outputs the argument as an error.
      output_debug: A callable that outputs the argument as debug.
      get_now: A callable that returns the current time.

    Yields:
      Entries for the tail session.
    """
    request = self.client.types.TailLogEntriesRequest()
    request.resource_names.extend(resource_names)
    request.filter = logs_filter
    self.tail_stub = gapic_util.MakeBidiRpc(self.client, self.client.logging.transport.tail_log_entries, initial_request=request)
    if buffer_window_seconds:
        request.buffer_window = datetime.timedelta(seconds=buffer_window_seconds)
    for entry in _StreamEntries(get_now, output_warning, output_error, output_debug, self.tail_stub):
        yield entry