import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _is_valid_directive_field(field):
    return any([field in _DISALLOW_DIRECTIVE, field in _ALLOW_DIRECTIVE, field in _USER_AGENT_DIRECTIVE, field in _SITEMAP_DIRECTIVE, field in _CRAWL_DELAY_DIRECTIVE, field in _REQUEST_RATE_DIRECTIVE, field in _HOST_DIRECTIVE])