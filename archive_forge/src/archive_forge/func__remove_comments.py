from __future__ import annotations
import re
from .nbbase import new_code_cell, new_notebook, new_text_cell, new_worksheet
from .rwbase import NotebookReader, NotebookWriter
def _remove_comments(self, lines):
    new_lines = []
    for line in lines:
        if line.startswith('#'):
            new_lines.append(line[2:])
        else:
            new_lines.append(line)
    text = '\n'.join(new_lines)
    text = text.strip('\n')
    return text