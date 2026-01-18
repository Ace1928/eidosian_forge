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

        Helper function to create a temporary directory.

        Returns
        -------
        contextmanager
            Context manager that returns the path to a temporary directory.
        