import warnings
from logging import getLogger
from scrapy import signals
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Response, TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.utils._compression import (
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.gz import gunzip
This middleware allows compressed (gzip, deflate) traffic to be
    sent/received from web sites