import io
from .. import ui
from ..ui import text as ui_text
class StringIOAsTTY(StringIOWithEncoding):
    """A helper class which makes a StringIO look like a terminal"""

    def isatty(self):
        return True