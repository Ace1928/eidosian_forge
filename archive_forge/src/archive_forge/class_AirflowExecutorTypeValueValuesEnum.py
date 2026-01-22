from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AirflowExecutorTypeValueValuesEnum(_messages.Enum):
    """The `airflowExecutorType` specifies the [executor](https://airflow.apa
    che.org/code.html?highlight=executor#executors) by which task instances
    are run on Airflow. If this field is unspecified, the
    `airflowExecutorType` defaults to `celery`.

    Values:
      AIRFLOW_EXECUTOR_TYPE_UNSPECIFIED: The Airflow executor type is
        unspecified.
      CELERY: The Celery executor will be used.
      KUBERNETES: The Kubernetes executor will be used.
    """
    AIRFLOW_EXECUTOR_TYPE_UNSPECIFIED = 0
    CELERY = 1
    KUBERNETES = 2