from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _CreateActionsText(text, igm_field, action_types):
    """Creates text presented at each wait operation for given IGM field.

  Args:
    text: the text associated with the field.
    igm_field: reference to a field in the Instance Group Manager.
    action_types: array with field values to be counted.
  Returns:
    A message with given field and action types count for IGM.
  """
    actions = []
    for action in action_types:
        action_count = getattr(igm_field, action, None) or 0
        if action_count > 0:
            actions.append('{0}: {1}'.format(action, action_count))
    return text + ', '.join(actions) if actions else ''