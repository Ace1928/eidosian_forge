import re
from traitlets import Dict
from .base import Preprocessor
def which_magic_language(self, source):
    """
        When a cell uses another language through a magic extension,
        the other language is returned.
        If no language magic is detected, this function returns None.

        Parameters
        ----------
        source: str
            Source code of the cell to highlight
        """
    m = self.re_magic_language.match(source)
    if m:
        return self.default_languages[m.group(1)]
    return None