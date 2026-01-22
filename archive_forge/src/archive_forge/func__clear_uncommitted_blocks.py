import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def _clear_uncommitted_blocks(conf: Config, url: str, metadata: Dict[str, str]) -> Optional['urllib3.BaseHTTPResponse']:
    req = Request(url=url, params=dict(comp='blocklist'), method='GET', success_codes=(200, 404))
    resp = execute_api_request(conf, req)
    if resp.status != 200:
        return
    result = xml.parse(resp.data, repeated_tags={'Block'})
    if result['BlockList']['CommittedBlocks'] is None:
        return
    blocks = result['BlockList']['CommittedBlocks']['Block']
    body = {'BlockList': {'Latest': [b['Name'] for b in blocks]}}
    headers: Dict[str, str] = {k: v for k, v in metadata.items() if k.startswith('x-ms-meta-')}
    for src, dst in RESPONSE_HEADER_TO_REQUEST_HEADER.items():
        if src in metadata:
            headers[dst] = metadata[src]
    req = Request(url=url, method='PUT', params=dict(comp='blocklist'), headers={**headers, 'If-Match': metadata['etag']}, data=body, success_codes=(201, 404, 412))
    return execute_api_request(conf, req)