import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
def assertErrorPage(self, status, message=None, pattern=''):
    """Compare the response body with a built in error page.

        The function will optionally look for the regexp pattern,
        within the exception embedded in the error page."""
    page = cherrypy._cperror.get_error_page(status, message=message)

    def esc(text):
        return re.escape(ntob(text))
    epage = re.escape(page)
    epage = epage.replace(esc('<pre id="traceback"></pre>'), esc('<pre id="traceback">') + b'(.*)' + esc('</pre>'))
    m = re.match(epage, self.body, re.DOTALL)
    if not m:
        self._handlewebError('Error page does not match; expected:\n' + page)
        return
    if pattern is None:
        if m and m.group(1):
            self._handlewebError('Error page contains traceback')
    elif m is None or not re.search(ntob(re.escape(pattern), self.encoding), m.group(1)):
        msg = 'Error page does not contain %s in traceback'
        self._handlewebError(msg % repr(pattern))