from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceStateValueValuesEnum(_messages.Enum):
    """Instances in which state should be returned. Valid options are: 'ALL',
    'RUNNING'. By default, it lists all instances.

    Values:
      ALL: Matches any status of the instances, running, non-running and
        others.
      RUNNING: Instance is in RUNNING state if it is running.
    """
    ALL = 0
    RUNNING = 1