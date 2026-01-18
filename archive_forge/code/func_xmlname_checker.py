import functools
import hashlib
import os
import posixpath
import re
import sys
import tempfile
import traceback
import warnings
from datetime import datetime
from importlib import import_module
from os import path
from time import mktime, strptime
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable, List,
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import ExtensionError, FiletypeNotFoundError, SphinxParallelError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold, colorize, strip_colors, term_width_line  # type: ignore
from sphinx.util.matching import patfilter  # noqa
from sphinx.util.nodes import (caption_ref_re, explicit_title_re,  # noqa
from sphinx.util.osutil import (SEP, copyfile, copytimes, ensuredir, make_filename,  # noqa
from sphinx.util.typing import PathMatcher
def xmlname_checker() -> Pattern:
    name_start_chars = [':', ['A', 'Z'], '_', ['a', 'z'], ['À', 'Ö'], ['Ø', 'ö'], ['ø', '˿'], ['Ͱ', 'ͽ'], ['Ϳ', '\u1fff'], ['\u200c', '\u200d'], ['⁰', '\u218f'], ['Ⰰ', '\u2fef'], ['、', '\ud7ff'], ['豈', '﷏'], ['ﷰ', '�'], ['𐀀', '\U000effff']]
    name_chars = ['\\-', '\\.', ['0', '9'], '·', ['̀', 'ͯ'], ['‿', '⁀']]

    def convert(entries: Any, splitter: str='|') -> str:
        results = []
        for entry in entries:
            if isinstance(entry, list):
                results.append('[%s]' % convert(entry, '-'))
            else:
                results.append(entry)
        return splitter.join(results)
    start_chars_regex = convert(name_start_chars)
    name_chars_regex = convert(name_chars)
    return re.compile('(%s)(%s|%s)*' % (start_chars_regex, start_chars_regex, name_chars_regex))