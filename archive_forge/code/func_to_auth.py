import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
def to_auth(self, client=None):
    """Returns the auth info object for this dsn."""
    return Auth(scheme=self.scheme, host=self.netloc, path=self.path, project_id=self.project_id, public_key=self.public_key, secret_key=self.secret_key, client=client)