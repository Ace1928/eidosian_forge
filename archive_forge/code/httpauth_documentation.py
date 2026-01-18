import warnings
from w3lib.http import basic_auth_header
from scrapy import signals
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.url import url_is_from_any_domain
Set Basic HTTP Authorization header
    (http_user and http_pass spider class attributes)