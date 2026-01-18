import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_with_method(self):

    class foo(object):

        def __init__(self):
            pass

        @deprecated(version='test')
        def bar(self):
            logger.warning('yeah')
    self.assertRegex(foo.bar.__doc__, '.. deprecated:: test\\n   This function \\(.*\\.foo\\.bar\\) has been deprecated')
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo().bar()
    self.assertIn('yeah', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertRegex(DEP_OUT.getvalue().replace('\n', ' '), 'DEPRECATED: This function \\(.*\\.foo\\.bar\\) has been deprecated.*\\(deprecated in test\\)')