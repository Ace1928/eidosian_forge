import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_with_custom_message(self):

    @deprecated('This is a custom message, too.', version='test')
    def foo(bar='yeah'):
        """Show that I am a good person.

            Because I document my public functions.

            """
        logger.warning(bar)
    self.assertIn('.. deprecated:: test\n   This is a custom message', foo.__doc__)
    self.assertIn('I am a good person.', foo.__doc__)
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo()
    self.assertIn('yeah', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertIn('DEPRECATED: This is a custom message', DEP_OUT.getvalue())
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo('custom')
    self.assertNotIn('yeah', FCN_OUT.getvalue())
    self.assertIn('custom', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertIn('DEPRECATED: This is a custom message', DEP_OUT.getvalue())