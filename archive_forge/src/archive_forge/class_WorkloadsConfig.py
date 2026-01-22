from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadsConfig(_messages.Message):
    """The Kubernetes workloads configuration for GKE cluster associated with
  the Cloud Composer environment. Supported for Cloud Composer environments in
  versions composer-2.*.*-airflow-*.*.* and newer.

  Fields:
    dagProcessor: Optional. Resources used by Airflow DAG processors. This
      field is supported for Cloud Composer environments in versions
      composer-3.*.*-airflow-*.*.* and newer.
    scheduler: Optional. Resources used by Airflow scheduler.
    schedulerCpu: Optional. CPU request and limit for Airflow scheduler.
    triggerer: Optional. Resources used by Airflow triggerers.
    webServer: Optional. Resources used by Airflow web server.
    worker: Optional. Resources used by Airflow workers.
    workerCpu: Optional. CPU request and limit for Airflow worker.
    workerMaxCount: Optional. Maximum number of workers for autoscaling.
    workerMinCount: Optional. Minimum number of workers for autoscaling.
  """
    dagProcessor = _messages.MessageField('DagProcessorResource', 1)
    scheduler = _messages.MessageField('SchedulerResource', 2)
    schedulerCpu = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    triggerer = _messages.MessageField('TriggererResource', 4)
    webServer = _messages.MessageField('WebServerResource', 5)
    worker = _messages.MessageField('WorkerResource', 6)
    workerCpu = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    workerMaxCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    workerMinCount = _messages.IntegerField(9, variant=_messages.Variant.INT32)