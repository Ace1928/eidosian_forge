import doctest
import logging
import re
from testpath import modified_env
class IPDocTestRunner(doctest.DocTestRunner, object):
    """Test runner that synchronizes the IPython namespace with test globals.
    """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        with modified_env({'COLUMNS': '80', 'LINES': '24'}):
            return super(IPDocTestRunner, self).run(test, compileflags, out, clear_globs)