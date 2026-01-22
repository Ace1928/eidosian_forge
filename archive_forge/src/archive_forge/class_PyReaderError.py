from __future__ import annotations
import re
from .nbbase import new_code_cell, new_notebook, new_text_cell, new_worksheet
from .rwbase import NotebookReader, NotebookWriter
class PyReaderError(Exception):
    """An error raised by the PyReader."""