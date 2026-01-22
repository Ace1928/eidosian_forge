import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def _CheckCookie(self):
    """Warn if cookie is not valid for at least one minute."""
    min_expire = time.time() + 60
    for cookie in self.cookie_jar:
        if cookie.domain == self.host and (not cookie.is_expired(min_expire)):
            break
    else:
        (print >> sys.stderr, '\nError: Machine system clock is incorrect.\n')