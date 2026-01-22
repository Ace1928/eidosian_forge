from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
class GCLLogTailer(v1_logs_util.TailerBase):
    """Helper class to tail logs from GCL, printing content as available."""
    CLOUDBUILD_BUCKET = 'cloudbuild'
    ALL_LOGS_VIEW = '_AllLogs'

    def __init__(self, project, location, log_filter, has_worker_pool, out=log.status):
        self.tailer = v1_logs_util.GetGCLLogTailer()
        self.log_filter = log_filter
        self.project = project
        self.location = location
        self.default_log_view = 'projects/{project_id}/locations/global/buckets/_Default/views/{view}'.format(project_id=self.project, view=self.ALL_LOGS_VIEW)
        self.workerpool_log_view = 'projects/{project_id}/locations/{location}/buckets/{bucket}/views/{view}'.format(project_id=self.project, location=self.location, bucket=self.CLOUDBUILD_BUCKET, view=self.ALL_LOGS_VIEW)
        self.has_worker_pool = has_worker_pool
        self.out = out
        self.buffer_window_seconds = 2

    @classmethod
    def FromFilter(cls, project, location, log_filter, has_worker_pool, out=log.out):
        """Build a GCLLogTailer from a log filter."""
        return cls(project=project, log_filter=log_filter, location=location, has_worker_pool=has_worker_pool, out=out)

    def Tail(self):
        """Tail the GCL logs and print any new bytes to the console."""
        if not self.tailer:
            return
        if self.has_worker_pool:
            resource_names = [self.workerpool_log_view]
        else:
            resource_names = [self.default_log_view]
        output_logs = self.tailer.TailLogs(resource_names, self.log_filter, buffer_window_seconds=self.buffer_window_seconds)
        self._PrintFirstLine(' REMOTE RUN OUTPUT ')
        for output in output_logs:
            text = self._ValidateScreenReader(output.text_payload)
            self._PrintLogLine(text)
        self._PrintLastLine(' RUN FINISHED; TRUNCATING OUTPUT LOGS ')
        return

    def Stop(self):
        """Stop log tailing."""
        time.sleep(self.buffer_window_seconds)
        if self.tailer:
            self.tailer.Stop()

    def Print(self):
        """Print GCL logs to the console."""
        if self.has_worker_pool:
            resource_names = [self.workerpool_log_view]
        else:
            resource_names = [self.default_log_view]
        output_logs = common.FetchLogs(log_filter=self.log_filter, order_by='asc', resource_names=resource_names)
        self._PrintFirstLine(' REMOTE RUN OUTPUT ')
        for output in output_logs:
            text = self._ValidateScreenReader(output.textPayload)
            self._PrintLogLine(text)
        self._PrintLastLine()