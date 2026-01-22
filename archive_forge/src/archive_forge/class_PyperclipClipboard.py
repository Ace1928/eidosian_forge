from __future__ import absolute_import, unicode_literals
import pyperclip
from prompt_toolkit.selection import SelectionType
from .base import Clipboard, ClipboardData
class PyperclipClipboard(Clipboard):
    """
    Clipboard that synchronizes with the Windows/Mac/Linux system clipboard,
    using the pyperclip module.
    """

    def __init__(self):
        self._data = None

    def set_data(self, data):
        assert isinstance(data, ClipboardData)
        self._data = data
        pyperclip.copy(data.text)

    def get_data(self):
        text = pyperclip.paste()
        if self._data and self._data.text == text:
            return self._data
        else:
            return ClipboardData(text=text, type=SelectionType.LINES if '\n' in text else SelectionType.LINES)