import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_language_metadata(self):
    """
        Build the language components of the User-Agent header string.

        Returns the Python version in a component with prefix "lang" and name
        "python". The Python implementation (e.g. CPython, PyPy) is returned as
        separate metadata component with prefix "md" and name "pyimpl".

        String representation of an example return value:
        ``lang/python#3.10.4 md/pyimpl#CPython``
        """
    lang_md = [UserAgentComponent('lang', 'python', self._python_version)]
    if self._python_implementation:
        lang_md.append(UserAgentComponent('md', 'pyimpl', self._python_implementation))
    return lang_md