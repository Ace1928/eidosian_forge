import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_with_doc_string(self):

    @deprecated(version='test')
    def foo(bar='yeah'):
        """Show that I am a good person.

            Because I document my public functions.

            """
        logger.warning(bar)
    self.assertRegex(foo.__doc__, 'I am a good person.\\s+Because I document my public functions.\\s+.. deprecated:: test\\n   This function \\(.*\\.foo\\) has been deprecated')
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo()
    self.assertIn('yeah', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertRegex(DEP_OUT.getvalue().replace('\n', ' '), 'DEPRECATED: This function \\(.*\\.foo\\) has been deprecated')
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo('custom')
    self.assertNotIn('yeah', FCN_OUT.getvalue())
    self.assertIn('custom', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertRegex(DEP_OUT.getvalue().replace('\n', ' '), 'DEPRECATED: This function \\(.*\\.foo\\) has been deprecated')