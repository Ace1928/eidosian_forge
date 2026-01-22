import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def _do_strip_docstring(self, docstring: str) -> Tuple[str, str]:
    """Return contents of docstring and opening quote type.

        Strips the docstring of its triple quotes, trailing white space,
        and line returns.  Determines type of docstring quote (either string,
        raw, or unicode) and returns the opening quotes, including the type
        identifier, with single quotes replaced by double quotes.

        Parameters
        ----------
        docstring: str
            The docstring, including the opening and closing triple quotes.

        Returns
        -------
        (docstring, open_quote) : tuple
            The docstring with the triple quotes removed.
            The opening quote type with single quotes replaced by double
            quotes.
        """
    docstring = docstring.strip()
    for quote in self.QUOTE_TYPES:
        if quote in self.RAW_QUOTE_TYPES + self.UCODE_QUOTE_TYPES and (docstring.startswith(quote) and docstring.endswith(quote[1:])):
            return (docstring.split(quote, 1)[1].rsplit(quote[1:], 1)[0].strip(), quote.replace("'", '"'))
        elif docstring.startswith(quote) and docstring.endswith(quote):
            return (docstring.split(quote, 1)[1].rsplit(quote, 1)[0].strip(), quote.replace("'", '"'))
    raise ValueError('docformatter only handles triple-quoted (single or double) strings')