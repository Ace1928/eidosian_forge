import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
class QuirkyTestingDAVRequestHandler(TestingDAVRequestHandler):
    """Various quirky/slightly off-spec behaviors.

    Used to test how gracefully we handle them.
    """
    delete_success_code = 200
    move_default_overwrite = False