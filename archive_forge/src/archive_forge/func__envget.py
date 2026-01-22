import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def _envget(self):
    try:
        return self.req_data.environ
    except AttributeError:
        return None