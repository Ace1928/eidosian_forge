from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def TransformShortStatus(r, undefined='UNKNOWN_ERROR'):
    """Returns a short description of the status of a logpoint or snapshot.

  Status will be one of ACTIVE, COMPLETED, or a short error description. If
  the status is an error, there will be additional information available in the
  status field of the object.

  Args:
    r: a JSON-serializable object
    undefined: Returns this value if the resource is not a valid status.

  Returns:
    One of ACTIVE, COMPLETED, or an error description.

  Example:
    `--format="table(id, location, short_status())"`:::
    Displays the short status in the third table problem.
  """
    short_status, _ = _TransformStatuses(r, undefined)
    return short_status