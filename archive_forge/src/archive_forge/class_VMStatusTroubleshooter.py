from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
class VMStatusTroubleshooter(ssh_troubleshooter.SshTroubleshooter):
    """Check VM status.

  Performance cpu, memory, disk status checking.

  Attributes:
    project: The project object.
    zone: str, the zone name.
    instance: The instance object.
  """

    def __init__(self, project, zone, instance):
        self.project = project
        self.zone = zone
        self.instance = instance
        self.monitoring_client = apis.GetClientInstance(_API_MONITORING_CLIENT_NAME, _API_MONITORING_VERSION_V3)
        self.monitoring_message = apis.GetMessagesModule(_API_MONITORING_CLIENT_NAME, _API_MONITORING_VERSION_V3)
        self.compute_client = apis.GetClientInstance(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_message = apis.GetMessagesModule(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.issues = {}

    def check_prerequisite(self):
        log.status.Print('---- Checking VM status ----')
        msg = "The Monitoring API is needed to check the VM's Status."
        prompt = "Is it OK to enable it and check the VM's Status?"
        cancel = 'Test skipped.'
        try:
            prompt_continue = console_io.PromptContinue(message=msg, prompt_string=prompt, cancel_on_no=True, cancel_string=cancel)
            self.skip_troubleshoot = not prompt_continue
        except OperationCancelledError:
            self.skip_troubleshoot = True
        if self.skip_troubleshoot:
            return
        enable_api.EnableService(self.project.name, MONITORING_API)

    def cleanup_resources(self):
        return

    def troubleshoot(self):
        if self.skip_troubleshoot:
            return
        self._CheckVMStatus()
        self._CheckCpuStatus()
        self._CheckDiskStatus()
        log.status.Print('VM status: {0} issue(s) found.\n'.format(len(self.issues)))
        for message in self.issues.values():
            log.status.Print(message)

    def _CheckVMStatus(self):
        if self.instance.status != self.compute_message.Instance.StatusValueValuesEnum.RUNNING:
            self.issues['vm_status'] = VM_STATUS_MESSAGE

    def _CheckCpuStatus(self):
        """Check cpu utilization."""
        cpu_utilizatian = self._GetCpuUtilization()
        if cpu_utilizatian > CPU_THRESHOLD:
            self.issues['cpu'] = CPU_MESSAGE

    def _GetCpuUtilization(self):
        """Get CPU utilization from Cloud Monitoring API."""
        for req_field, mapped_param in _CUSTOM_JSON_FIELD_MAPPINGS.items():
            encoding.AddCustomJsonFieldMapping(self.monitoring_message.MonitoringProjectsTimeSeriesListRequest, req_field, mapped_param)
        request = self._CreateTimeSeriesListRequest(CPU_METRICS)
        response = self.monitoring_client.projects_timeSeries.List(request=request)
        if response.timeSeries:
            points = response.timeSeries[0].points
            return sum((point.value.doubleValue for point in points)) / len(points)
        return 0.0

    def _CheckDiskStatus(self):
        sc_log = ssh_troubleshooter_utils.GetSerialConsoleLog(self.compute_client, self.compute_message, self.instance.name, self.project.name, self.zone)
        if ssh_troubleshooter_utils.SearchPatternErrorInLog(DISK_ERROR_PATTERN, sc_log):
            self.issues['disk'] = DISK_MESSAGE.format(self.instance.disks[0].source)

    def _CreateTimeSeriesListRequest(self, metrics_type):
        """Create a MonitoringProjectsTimeSeriesListRequest.

    Args:
      metrics_type: str, https://cloud.google.com/monitoring/api/metrics

    Returns:
      MonitoringProjectsTimeSeriesListRequest, input message for
      ProjectsTimeSeriesService List method.
    """
        current_time = datetime.datetime.utcnow()
        tp_proto_end_time = timestamp_pb2.Timestamp()
        tp_proto_end_time.FromDatetime(current_time)
        tp_proto_start_time = timestamp_pb2.Timestamp()
        tp_proto_start_time.FromDatetime(current_time - datetime.timedelta(seconds=600))
        return self.monitoring_message.MonitoringProjectsTimeSeriesListRequest(name='projects/{project}'.format(project=self.project.name), filter=FILTER_TEMPLATE.format(metrics_type=metrics_type, instance_name=self.instance.name), interval_endTime=tp_proto_end_time.ToJsonString(), interval_startTime=tp_proto_start_time.ToJsonString())