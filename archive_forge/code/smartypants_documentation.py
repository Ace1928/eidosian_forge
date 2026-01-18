import re
import warnings
from typing import Generator, Iterable, Tuple
from docutils.utils import smartquotes
from sphinx.deprecation import RemovedInSphinx60Warning
Return iterator that "educates" the items of `text_tokens`.

    This is modified to intercept the ``attr='2'`` as it was used by the
    Docutils 0.13.1 SmartQuotes transform in a hard coded way. Docutils 0.14
    uses ``'qDe'``` and is configurable, and its choice is backported here
    for use by Sphinx with earlier Docutils releases. Similarly ``'1'`` is
    replaced by ``'qde'``.

    Use ``attr='qDbe'``, resp. ``'qdbe'`` to recover Docutils effect of ``'2'``,
    resp. ``'1'``.

    refs: https://sourceforge.net/p/docutils/mailman/message/35869025/
    