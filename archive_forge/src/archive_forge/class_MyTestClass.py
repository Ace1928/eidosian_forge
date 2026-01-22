import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
class MyTestClass(HasTraits):
    """
    Class-level docstring.
    """
    bar = Int(42, desc=' First line\n\n        The answer to\n        Life,\n        the Universe,\n\n        and Everything.\n    ')