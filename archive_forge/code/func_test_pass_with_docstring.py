import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_pass_with_docstring(self):
    """ The test that always passes.

                    This test case has a PEP 257 conformant docstring,
                    with its first line being a brief synopsis and the
                    rest of the docstring explaining that this test
                    does nothing but pass unconditionally.

                    """
    pass