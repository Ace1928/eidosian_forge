from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeStateValueValuesEnum(_messages.Enum):
    """Output only. State of the change.

    Values:
      LINE_ITEM_CHANGE_STATE_UNSPECIFIED: Sentinel value. Do not use.
      LINE_ITEM_CHANGE_STATE_PENDING_APPROVAL: Change is in this state when a
        change is initiated and waiting for partner approval. This state is
        only applicable for pending change.
      LINE_ITEM_CHANGE_STATE_APPROVED: Change is in this state after it's
        approved by the partner or auto-approved but before it takes effect.
        The change can be overwritten or cancelled depending on the new line
        item info property (pending Private Offer change cannot be cancelled
        and can only be overwritten by another Private Offer). This state is
        only applicable for pending change.
      LINE_ITEM_CHANGE_STATE_COMPLETED: Change is in this state after it's
        been activated. This state is only applicable for change in history.
      LINE_ITEM_CHANGE_STATE_REJECTED: Change is in this state if it was
        rejected by the partner. This state is only applicable for change in
        history.
      LINE_ITEM_CHANGE_STATE_ABANDONED: Change is in this state if it was
        abandoned by the user. This state is only applicable for change in
        history.
      LINE_ITEM_CHANGE_STATE_ACTIVATING: Change is in this state if it's
        currently being provisioned downstream. The change can't be
        overwritten or cancelled when it's in this state. This state is only
        applicable for pending change.
    """
    LINE_ITEM_CHANGE_STATE_UNSPECIFIED = 0
    LINE_ITEM_CHANGE_STATE_PENDING_APPROVAL = 1
    LINE_ITEM_CHANGE_STATE_APPROVED = 2
    LINE_ITEM_CHANGE_STATE_COMPLETED = 3
    LINE_ITEM_CHANGE_STATE_REJECTED = 4
    LINE_ITEM_CHANGE_STATE_ABANDONED = 5
    LINE_ITEM_CHANGE_STATE_ACTIVATING = 6