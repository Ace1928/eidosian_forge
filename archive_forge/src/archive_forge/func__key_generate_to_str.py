import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
def _key_generate_to_str(s):
    try:
        return str(s)
    except UnicodeEncodeError:
        return s.encode('utf-8')