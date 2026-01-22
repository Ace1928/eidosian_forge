import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
class GoogleAuthMockHttp(MockHttp):
    """
    Mock HTTP Class for Google Auth Connections.
    """
    json_hdr = {'content-type': 'application/json; charset=UTF-8'}

    def _o_oauth2_token(self, method, url, body, headers):
        if 'code' in body:
            body = json.dumps(STUB_IA_TOKEN)
        elif 'refresh_token' in body:
            body = json.dumps(STUB_REFRESH_TOKEN)
        else:
            body = json.dumps(STUB_TOKEN)
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])