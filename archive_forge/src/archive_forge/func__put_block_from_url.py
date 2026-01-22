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
def _put_block_from_url(conf: Config, src: str, start: int, size: int, dst: str, block_id: str) -> None:
    src_account, src_container, src_blob = split_path(src)
    dst_account, dst_container, dst_blob = split_path(dst)
    src_url = build_url(src_account, '/{container}/{blob}', container=src_container, blob=src_blob)
    dst_url = build_url(dst_account, '/{container}/{blob}', container=dst_container, blob=dst_blob)

    def build_req() -> Request:
        sas_token = sas_token_manager.get_token(conf=conf, key=(src_account, src_container))
        if sas_token is None:
            copy_src_url = src_url
        else:
            copy_src_url, _ = generate_signed_url(key=sas_token, url=src_url)
        req = Request(url=dst_url, method='PUT', params=dict(comp='block', blockid=block_id), headers={'Content-Length': '0', 'x-ms-copy-source': copy_src_url, 'x-ms-source-range': common.calc_range(start=start, end=start + size)}, success_codes=(201, 404, INVALID_HOSTNAME_STATUS))
        return create_api_request(req, auth=access_token_manager.get_token(conf=conf, key=(dst_account, dst_container)))
    resp = common.execute_request(conf, build_req)
    if resp.status == 404:
        raise FileNotFoundError(f"Source file/container or destination container not found: src='{src}' dst='{dst}'")
    elif resp.status == INVALID_HOSTNAME_STATUS:
        raise FileNotFoundError(f"Source container or destination container not found: src='{src}' dst='{dst}'")