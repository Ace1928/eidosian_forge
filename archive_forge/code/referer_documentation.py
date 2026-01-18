import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url

        Determine Referrer-Policy to use from a parent Response (or URL),
        and a Request to be sent.

        - if a valid policy is set in Request meta, it is used.
        - if the policy is set in meta but is wrong (e.g. a typo error),
          the policy from settings is used
        - if the policy is not set in Request meta,
          but there is a Referrer-policy header in the parent response,
          it is used if valid
        - otherwise, the policy from settings is used.
        