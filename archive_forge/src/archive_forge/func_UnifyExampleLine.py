from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def UnifyExampleLine(example_doc, pos):
    """Returns the example command line at pos in one single line.

  pos is the starting point of an example (starting with "$ ").
  This function removes "
" and "" and redundant spaces in the example line.
  The resulted example should be in one single line.

  Args:
    example_doc: str, Example section of the help text.
    pos: int, Position to start. pos will be the starting position of an
     example line.

  Returns:
    normalized example command, next starting position to search
  """
    pat_match_next_command = re.compile('\\$\\s+(.+?)(\\n +\\$\\s+)', re.DOTALL)
    pat_match_empty_line_after_command = re.compile('\\$\\s+(.+?)(\\n\\s*\\n|\\n\\+\\n)', re.DOTALL)
    match_next_command = pat_match_next_command.match(example_doc, pos)
    match_empty_line_after_command = pat_match_empty_line_after_command.match(example_doc, pos)
    if not match_next_command and (not match_empty_line_after_command):
        new_doc = example_doc.rstrip()
        pat = re.compile('\\$\\s+(.+)', re.DOTALL)
        match = pat.match(new_doc, pos)
        example = match.group(1)
        pat = re.compile('\\\\\\n\\s*')
        example = pat.sub('', example)
        example = RemoveSpacesLineBreaksFromExample(example)
        return ('$ ' + example, len(new_doc))
    elif match_next_command and match_empty_line_after_command:
        if len(match_next_command.group(1)) > len(match_empty_line_after_command.group(1)):
            selected_match = match_empty_line_after_command
        else:
            selected_match = match_next_command
    else:
        selected_match = match_next_command if match_next_command else match_empty_line_after_command
    example = selected_match.group(1)
    pat = re.compile('\\\\\\n\\s*')
    example = pat.sub('', example)
    example = RemoveSpacesLineBreaksFromExample(example)
    next_pos = selected_match.end(1)
    return ('$ ' + example, next_pos)