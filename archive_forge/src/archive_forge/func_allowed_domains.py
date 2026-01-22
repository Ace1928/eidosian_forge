import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def allowed_domains(self):
    """Return None, or the sequence of allowed domains (as a tuple)."""
    return self._allowed_domains