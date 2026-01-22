from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsedWidgetsValueListEntryValuesEnum(_messages.Enum):
    """UsedWidgetsValueListEntryValuesEnum enum type.

    Values:
      WIDGET_TYPE_UNSPECIFIED: The default widget set.
      DATE_PICKER: The date picker.
      STYLED_BUTTONS: Styled buttons include filled buttons and deactivated
        buttons.
      PERSISTENT_FORMS: Persistent forms allow persisting form values during
        actions.
      FIXED_FOOTER: Fixed footer in a card.
      UPDATE_SUBJECT_AND_RECIPIENTS: Update the subject and recipients of a
        draft.
      GRID_WIDGET: The grid widget.
      ADDON_COMPOSE_UI_ACTION: A Gmail add-on action that applies to the add-
        on compose UI.
    """
    WIDGET_TYPE_UNSPECIFIED = 0
    DATE_PICKER = 1
    STYLED_BUTTONS = 2
    PERSISTENT_FORMS = 3
    FIXED_FOOTER = 4
    UPDATE_SUBJECT_AND_RECIPIENTS = 5
    GRID_WIDGET = 6
    ADDON_COMPOSE_UI_ACTION = 7