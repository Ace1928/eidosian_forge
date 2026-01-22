from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionEnvironmentValueValuesEnum(_messages.Enum):
    """Optional. The execution environment being used to host this Task.

    Values:
      EXECUTION_ENVIRONMENT_UNSPECIFIED: Unspecified
      EXECUTION_ENVIRONMENT_GEN1: Uses the First Generation environment.
      EXECUTION_ENVIRONMENT_GEN2: Uses Second Generation environment.
    """
    EXECUTION_ENVIRONMENT_UNSPECIFIED = 0
    EXECUTION_ENVIRONMENT_GEN1 = 1
    EXECUTION_ENVIRONMENT_GEN2 = 2