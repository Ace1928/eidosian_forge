from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.resource import custom_printer_base as cp
class ExecutionPrinter(cp.CustomPrinterBase):
    """Prints the run Execution in a custom human-readable format.

  Format specific to Cloud Run jobs. Only available on Cloud Run commands
  that print executions.
  """

    @staticmethod
    def TransformSpec(record):
        limits = container_util.GetLimits(record.template)
        breakglass_value = k8s_util.GetBinAuthzBreakglass(record)
        return cp.Labeled([('Image', record.template.image), ('Tasks', record.spec.taskCount), ('Command', ' '.join(record.template.container.command)), ('Args', ' '.join(record.template.container.args)), ('Binary Authorization', k8s_util.GetBinAuthzPolicy(record)), ('Breakglass Justification', ' ' if breakglass_value == '' else breakglass_value), ('Memory', limits['memory']), ('CPU', limits['cpu']), ('Task Timeout', FormatDurationShort(record.template.spec.timeoutSeconds) if record.template.spec.timeoutSeconds else None), ('Max Retries', '{}'.format(record.template.spec.maxRetries) if record.template.spec.maxRetries is not None else None), ('Parallelism', record.parallelism), ('Service account', record.template.service_account), ('Env vars', container_util.GetUserEnvironmentVariables(record.template)), ('Secrets', container_util.GetSecrets(record.template.container)), ('VPC access', k8s_util.GetVpcNetwork(record.annotations)), ('SQL connections', k8s_util.GetCloudSqlInstances(record.annotations)), ('Volume Mounts', container_util.GetVolumeMounts(record.template.container)), ('Volumes', container_util.GetVolumes(record.template))])

    @staticmethod
    def TransformStatus(record):
        if record.status is None:
            return ''
        lines = []
        if record.ready_condition['status'] is None:
            lines.append('{} currently running'.format(_PluralizedWord('task', record.status.runningCount)))
        lines.append('{} completed successfully'.format(_PluralizedWord('task', record.status.succeededCount)))
        if record.status.failedCount is not None and record.status.failedCount > 0:
            lines.append('{} failed to complete'.format(_PluralizedWord('task', record.status.failedCount)))
        if record.status.cancelledCount is not None and record.status.cancelledCount > 0:
            lines.append('{} cancelled'.format(_PluralizedWord('task', record.status.cancelledCount)))
        if record.status.completionTime is not None and record.creation_timestamp is not None:
            lines.append('Elapsed time: ' + ExecutionPrinter._elapsedTime(record.creation_timestamp, record.status.completionTime))
        if record.status.logUri is not None:
            lines.append(' ')
            lines.append('Log URI: {}'.format(record.status.logUri))
        return cp.Lines(lines)

    @staticmethod
    def _elapsedTime(start, end):
        duration = datetime.timedelta(seconds=time_util.Strptime(end) - time_util.Strptime(start)).seconds
        hours = duration // 3600
        duration = duration % 3600
        minutes = duration // 60
        seconds = duration % 60
        if hours > 0:
            return '{} and {}'.format(_PluralizedWord('hour', hours), _PluralizedWord('minute', minutes))
        if minutes > 0:
            return '{} and {}'.format(_PluralizedWord('minute', minutes), _PluralizedWord('second', seconds))
        return _PluralizedWord('second', seconds)

    @staticmethod
    def _formatOutput(record):
        output = []
        header = k8s_util.BuildHeader(record)
        status = ExecutionPrinter.TransformStatus(record)
        labels = k8s_util.GetLabels(record.labels)
        spec = ExecutionPrinter.TransformSpec(record)
        ready_message = k8s_util.FormatReadyMessage(record)
        if header:
            output.append(header)
        if status:
            output.append(status)
        output.append(' ')
        if labels:
            output.append(labels)
            output.append(' ')
        if spec:
            output.append(spec)
        if ready_message:
            output.append(ready_message)
        return output

    def Transform(self, record):
        """Transform a job into the output structure of marker classes."""
        fmt = cp.Lines(ExecutionPrinter._formatOutput(record))
        return fmt