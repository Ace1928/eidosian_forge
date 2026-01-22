import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def _prop(ns, name, value=None):
    if value is None:
        return '<{}:{}/>'.format(ns, name)
    else:
        return '<{}:{}>{}</{}:{}>'.format(ns, name, value, ns, name)