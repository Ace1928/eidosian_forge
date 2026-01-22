import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@staticmethod
def _check_redirects(result: requests.Response) -> None:
    if not result.history:
        return
    for item in result.history:
        if item.status_code not in (301, 302):
            continue
        if item.request.method == 'GET':
            continue
        target = item.headers.get('location')
        raise gitlab.exceptions.RedirectError(REDIRECT_MSG.format(status_code=item.status_code, reason=item.reason, source=item.url, target=target))