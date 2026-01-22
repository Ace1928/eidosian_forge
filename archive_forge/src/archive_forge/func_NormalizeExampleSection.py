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
def NormalizeExampleSection(doc):
    """Removes line breaks and extra spaces in example commands.

  In command implementation, some example commands were manually broken into
  multiple lines with or without "". This function removes these line
  breaks and let ExampleCommandLineSplitter to split the long commands
  centrally.

  This function will not change example commands in the following situations:

  1. If the command is in a code block, surrounded with ```sh...```.
  2. If the values are within a quote (single or double quote).

  Args:
    doc: str, help text to process.

  Returns:
    Modified help text.
  """
    example_sec_until_next_sec = re.compile('^## EXAMPLES\\n(.+?)(\\n+## )', flags=re.M | re.DOTALL)
    example_sec_until_end = re.compile('^## EXAMPLES\\n(.+)', flags=re.M | re.DOTALL)
    match_example_sec = example_sec_until_next_sec.search(doc)
    match_example_sec_to_end = example_sec_until_end.search(doc)
    if not match_example_sec and (not match_example_sec_to_end):
        return doc
    elif match_example_sec:
        selected_match = match_example_sec
    else:
        selected_match = match_example_sec_to_end
    doc_before_examples = doc[:selected_match.start(1)]
    example_section = doc[selected_match.start(1):selected_match.end(1)]
    doc_after_example = doc[selected_match.end(1):]
    pat_example_line = re.compile('^ *(\\$ .*)$', re.M)
    pat_code_block = re.compile('^ *```sh(.+?```)', re.M | re.DOTALL)
    pos = 0
    res = ''
    while True:
        match_example_line = pat_example_line.search(example_section, pos)
        match_code_block = pat_code_block.search(example_section, pos)
        if not match_code_block and (not match_example_line):
            break
        elif match_code_block and match_example_line:
            if match_code_block.start(1) > match_example_line.start(1):
                example, next_pos = UnifyExampleLine(example_section, match_example_line.start(1))
                res += example_section[pos:match_example_line.start(1)] + example
                pos = next_pos
            else:
                res += example_section[pos:match_code_block.end(1)]
                pos = match_code_block.end(1)
        elif match_code_block:
            res += example_section[pos:match_code_block.end(1)]
            pos = match_code_block.end(1)
        else:
            example, next_pos = UnifyExampleLine(example_section, match_example_line.start(1))
            res += example_section[pos:match_example_line.start(1)] + example
            pos = next_pos
    return doc_before_examples + (res + example_section[pos:]) + doc_after_example