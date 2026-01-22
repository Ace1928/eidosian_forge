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
def _get_sas_token(conf: Config, key: Any) -> Tuple[Any, float]:
    auth = access_token_manager.get_token(conf, key=key)
    if auth[0] == ANONYMOUS:
        return (None, time.time() + SAS_TOKEN_EXPIRATION_SECONDS)
    account, container = key

    def build_req() -> Request:
        now = datetime.datetime.utcnow()
        start = (now + datetime.timedelta(hours=-1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        expiration = now + datetime.timedelta(days=6)
        expiry = expiration.strftime('%Y-%m-%dT%H:%M:%SZ')
        req = Request(url=f'https://{account}.blob.core.windows.net/', method='POST', params=dict(restype='service', comp='userdelegationkey'), data={'KeyInfo': {'Start': start, 'Expiry': expiry}}, success_codes=(200, 403))
        auth = access_token_manager.get_token(conf, key=key)
        if auth[0] != OAUTH_TOKEN:
            raise Error(f'Only OAuth tokens can be used to get SAS tokens. You should set the Storage Blob Data Reader or Storage Blob Data Contributor IAM role. You can run `az storage blob list --auth-mode login --account-name {account} --container {container}` to confirm that the missing role is the issue.')
        return create_api_request(req, auth=auth)
    resp = common.execute_request(conf, build_req)
    if resp.status == 403:
        raise Error(f'You do not have permission to generate an SAS token for account {account}. Try setting the Storage Blob Delegator or Storage Blob Data Contributor IAM role at the account level.')
    out = xml.parse(resp.data)
    t = time.time() + SAS_TOKEN_EXPIRATION_SECONDS
    return (out['UserDelegationKey'], t)