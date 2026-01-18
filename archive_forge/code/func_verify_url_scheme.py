import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase
def verify_url_scheme(url):
    """Check url for scheme and insert https if none found."""
    parsed = urlparse(url)
    if parsed.scheme == '' and parsed.netloc == '':
        parsed = urlparse('//' + url)._replace(scheme='https')
    return parsed.geturl()