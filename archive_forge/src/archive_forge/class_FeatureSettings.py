from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureSettings(_messages.Message):
    """The feature specific settings to be used in the application. These
  define behaviors that are user configurable.

  Fields:
    splitHealthChecks: Boolean value indicating if split health checks should
      be used instead of the legacy health checks. At an app.yaml level, this
      means defaulting to 'readiness_check' and 'liveness_check' values
      instead of 'health_check' ones. Once the legacy 'health_check' behavior
      is deprecated, and this value is always true, this setting can be
      removed.
    useContainerOptimizedOs: If true, use Container-Optimized OS
      (https://cloud.google.com/container-optimized-os/) base image for VMs,
      rather than a base Debian image.
  """
    splitHealthChecks = _messages.BooleanField(1)
    useContainerOptimizedOs = _messages.BooleanField(2)