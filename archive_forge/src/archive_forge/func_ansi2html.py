import re
import markupsafe
def ansi2html(text):
    """
    Convert ANSI colors to HTML colors.

    Parameters
    ----------
    text : unicode
        Text containing ANSI colors to convert to HTML

    """
    text = markupsafe.escape(text)
    return _ansi2anything(text, _htmlconverter)