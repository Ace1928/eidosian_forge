from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
def _GetNonWorkerPoolLogFilter(self, create_time, run_id, region):
    return 'timestamp>="{timestamp}" AND labels.location="{region}" AND labels.run_name={run_id}'.format(timestamp=create_time, region=region, run_id=run_id)