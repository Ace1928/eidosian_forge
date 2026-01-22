from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimePoliciesValueListEntryValuesEnum(_messages.Enum):
    """RuntimePoliciesValueListEntryValuesEnum enum type.

    Values:
      CREATE: The action will only be fired during create.
      DELETE: The action will only be fired when the action is removed from
        the deployment.
      UPDATE_ON_CHANGE: The action will fire during create, and if there is
        any changes on the inputs.
      UPDATE_ALWAYS: The action will fire during create, and every time there
        is an update to the deployment.
    """
    CREATE = 0
    DELETE = 1
    UPDATE_ON_CHANGE = 2
    UPDATE_ALWAYS = 3