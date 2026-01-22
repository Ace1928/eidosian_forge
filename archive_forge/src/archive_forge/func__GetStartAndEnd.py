from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def _GetStartAndEnd(match, cut_points, length_per_snippet):
    """Helper function to get start and end of single snippet that matches text.

  Gets a snippet of length length_per_snippet with the match object
  in the middle.
  Cuts at the first cut point (if available, else cuts at any char)
  within 1/2 the length of the start of the match object.
  Then cuts at the last cut point within
  the desired length (if available, else cuts at any point).
  Then moves start back if there is extra room at the beginning.

  Args:
    match: re.match object.
    cut_points: [int], indices of each cut char, plus start and
        end index of full string. Must be sorted.
        (The characters at cut_points are skipped.)
    length_per_snippet: int, max length of snippet to be returned

  Returns:
    (int, int) 2-tuple with start and end index of the snippet
  """
    max_length = cut_points[-1] if cut_points else 0
    match_start = match.start() if match else 0
    match_end = match.end() if match else 0
    start = 0
    if match_start > 0.5 * length_per_snippet:
        for c in cut_points:
            if c >= match_start - 0.5 * length_per_snippet and c < match_start:
                start = c + 1
                break
        start = int(max(match_start - 0.5 * length_per_snippet, start))
    end = match_end
    for c in cut_points:
        if end < c <= start + length_per_snippet:
            end = c
        elif c > start + length_per_snippet:
            break
    if end == match_end:
        end = max(min(max_length, start + length_per_snippet), end)
    if end == max_length:
        for c in cut_points:
            if end - c <= length_per_snippet + 1 and c < start:
                start = c + 1
                break
    return TextSlice(start, end)