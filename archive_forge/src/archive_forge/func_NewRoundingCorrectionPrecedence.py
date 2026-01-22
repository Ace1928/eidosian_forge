from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def NewRoundingCorrectionPrecedence(key_and_percent):
    """Returns object that sorts in the order we correct traffic rounding errors.

  The caller specifies explicit traffic percentages for some revisions and
  this module scales traffic for remaining revisions that are already
  serving traffic up or down to assure that 100% of traffic is assigned.
  This scaling can result in non integrer percentages that Cloud Run
  does not supprt. We correct by:
    - Trimming the decimal part of float_percent, int(float_percent)
    - Adding an extra 1 percent traffic to enough revisions that have
      had their traffic reduced to get us to 100%

  The returned value sorts in the order we correct revisions:
    1) Revisions with a bigger loss due are corrected before revisions with
       a smaller loss. Since 0 <= loss < 1 we sort by the value:  1 - loss.
    2) In the case of ties revisions with less traffic are corrected before
       revisions with more traffic.
    3) In case of a tie revisions with a smaller key are corrected before
       revisions with a larger key.

  Args:
    key_and_percent: tuple with (key, float_percent)

  Returns:
    An value that sorts with respect to values returned for
    other revisions in the order we correct for rounding
    errors.
  """
    key, float_percent = key_and_percent
    return [1 - (float_percent - int(float_percent)), float_percent, SortKeyFromKey(key)]