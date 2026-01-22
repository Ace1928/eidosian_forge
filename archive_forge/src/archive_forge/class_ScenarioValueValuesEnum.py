from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScenarioValueValuesEnum(_messages.Enum):
    """Output only. The scenario when the preflight checks were run.

    Values:
      SCENARIO_UNSPECIFIED: Default value. This value is unused.
      CREATE: The validation check occurred during a create flow.
      UPDATE: The validation check occurred during an update flow.
    """
    SCENARIO_UNSPECIFIED = 0
    CREATE = 1
    UPDATE = 2