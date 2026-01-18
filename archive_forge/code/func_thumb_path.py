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
def thumb_path(self, request, thumb_id, response=None, info=None, *, item=None):
    thumb_guid = hashlib.sha1(to_bytes(request.url)).hexdigest()
    return f'thumbs/{thumb_id}/{thumb_guid}.jpg'