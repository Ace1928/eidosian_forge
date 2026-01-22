from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResilienceModeValueValuesEnum(_messages.Enum):
    """Optional. Resilience mode of the Cloud Composer Environment. This
    field is supported for Cloud Composer environments in versions
    composer-2.2.0-airflow-*.*.* and newer.

    Values:
      RESILIENCE_MODE_UNSPECIFIED: Default mode doesn't change environment
        parameters.
      HIGH_RESILIENCE: Enabled High Resilience mode, including Cloud SQL HA.
    """
    RESILIENCE_MODE_UNSPECIFIED = 0
    HIGH_RESILIENCE = 1