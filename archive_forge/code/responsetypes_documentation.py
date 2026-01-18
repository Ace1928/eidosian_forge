from io import StringIO
from mimetypes import MimeTypes
from pkgutil import get_data
from typing import Dict, Mapping, Optional, Type, Union
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode
Guess the most appropriate Response class based on
        the given arguments.