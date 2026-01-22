import functools
import hashlib
import warnings
from contextlib import suppress
from io import BytesIO
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem, NotConfigured, ScrapyDeprecationWarning
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.files import FileException, FilesPipeline
from scrapy.settings import Settings
from scrapy.utils.misc import md5sum
from scrapy.utils.python import get_func_args, to_bytes
class NoimagesDrop(DropItem):
    """Product with no images exception"""

    def __init__(self, *args, **kwargs):
        warnings.warn('The NoimagesDrop class is deprecated', category=ScrapyDeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)