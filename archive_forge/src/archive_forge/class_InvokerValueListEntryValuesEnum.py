from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InvokerValueListEntryValuesEnum(_messages.Enum):
    """InvokerValueListEntryValuesEnum enum type.

    Values:
      INVOKER_UNSPECIFIED: Unspecified.
      USER: The action is user-driven (e.g. creating a rollout manually via a
        gcloud create command).
      DEPLOY_AUTOMATION: Automated action by Cloud Deploy.
    """
    INVOKER_UNSPECIFIED = 0
    USER = 1
    DEPLOY_AUTOMATION = 2