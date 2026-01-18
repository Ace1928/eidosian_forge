import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_unable_encode_path(self):
    err = errors.UnableEncodePath('foo', 'executable')
    self.assertEqual("Unable to encode executable path 'foo' in user encoding " + osutils.get_user_encoding(), str(err))