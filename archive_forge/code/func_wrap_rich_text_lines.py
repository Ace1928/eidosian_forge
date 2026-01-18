import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def wrap_rich_text_lines(inp, cols):
    """Wrap RichTextLines according to maximum number of columns.

  Produces a new RichTextLines object with the text lines, font_attr_segs and
  annotations properly wrapped. This ought to be used sparingly, as in most
  cases, command handlers producing RichTextLines outputs should know the
  screen/panel width via the screen_info kwarg and should produce properly
  length-limited lines in the output accordingly.

  Args:
    inp: Input RichTextLines object.
    cols: Number of columns, as an int.

  Returns:
    1) A new instance of RichTextLines, with line lengths limited to cols.
    2) A list of new (wrapped) line index. For example, if the original input
      consists of three lines and only the second line is wrapped, and it's
      wrapped into two lines, this return value will be: [0, 1, 3].
  Raises:
    ValueError: If inputs have invalid types.
  """
    new_line_indices = []
    if not isinstance(inp, RichTextLines):
        raise ValueError('Invalid type of input screen_output')
    if not isinstance(cols, int):
        raise ValueError('Invalid type of input cols')
    out = RichTextLines([])
    row_counter = 0
    for i, line in enumerate(inp.lines):
        new_line_indices.append(out.num_lines())
        if i in inp.annotations:
            out.annotations[row_counter] = inp.annotations[i]
        if len(line) <= cols:
            out.lines.append(line)
            if i in inp.font_attr_segs:
                out.font_attr_segs[row_counter] = inp.font_attr_segs[i]
            row_counter += 1
        else:
            wlines = []
            osegs = []
            if i in inp.font_attr_segs:
                osegs = inp.font_attr_segs[i]
            idx = 0
            while idx < len(line):
                if idx + cols > len(line):
                    rlim = len(line)
                else:
                    rlim = idx + cols
                wlines.append(line[idx:rlim])
                for seg in osegs:
                    if seg[0] < rlim and seg[1] >= idx:
                        if seg[0] >= idx:
                            lb = seg[0] - idx
                        else:
                            lb = 0
                        if seg[1] < rlim:
                            rb = seg[1] - idx
                        else:
                            rb = rlim - idx
                        if rb > lb:
                            wseg = (lb, rb, seg[2])
                            if row_counter not in out.font_attr_segs:
                                out.font_attr_segs[row_counter] = [wseg]
                            else:
                                out.font_attr_segs[row_counter].append(wseg)
                idx += cols
                row_counter += 1
            out.lines.extend(wlines)
    for key in inp.annotations:
        if not isinstance(key, int):
            out.annotations[key] = inp.annotations[key]
    return (out, new_line_indices)