from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def MonitoredProjectName(self, metrics_scope_ref, monitored_project_ref):
    return self.MetricsScopeName(metrics_scope_ref) + '/projects/' + monitored_project_ref.Name()