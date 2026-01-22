import base64
import functools
import hashlib
import logging
import mimetypes
import os
import time
from collections import defaultdict
from contextlib import suppress
from ftplib import FTP
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import DefaultDict, Optional, Set, Union
from urllib.parse import urlparse
from itemadapter import ItemAdapter
from twisted.internet import defer, threads
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.media import MediaPipeline
from scrapy.settings import Settings
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.datatypes import CaseInsensitiveDict
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import md5sum
from scrapy.utils.python import to_bytes
from scrapy.utils.request import referer_str
def _onsuccess(result):
    if not result:
        return
    last_modified = result.get('last_modified', None)
    if not last_modified:
        return
    age_seconds = time.time() - last_modified
    age_days = age_seconds / 60 / 60 / 24
    if age_days > self.expires:
        return
    referer = referer_str(request)
    logger.debug('File (uptodate): Downloaded %(medianame)s from %(request)s referred in <%(referer)s>', {'medianame': self.MEDIA_NAME, 'request': request, 'referer': referer}, extra={'spider': info.spider})
    self.inc_stats(info.spider, 'uptodate')
    checksum = result.get('checksum', None)
    return {'url': request.url, 'path': path, 'checksum': checksum, 'status': 'uptodate'}