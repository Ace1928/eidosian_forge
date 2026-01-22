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
def _get_subscription_ids(conf: Config, auth: Tuple[str, str]) -> List[str]:
    url = 'https://management.azure.com/subscriptions'
    params = {'api-version': '2020-01-01'}
    result = []
    while True:

        def build_req() -> Request:
            req = Request(method='GET', url=url, params=params)
            return create_api_request(req, auth=auth)
        resp = common.execute_request(conf, build_req)
        data = json.loads(resp.data)
        result.extend([item['subscriptionId'] for item in data['value']])
        if 'nextLink' not in data:
            return result
        url = data['nextLink']
        params = None