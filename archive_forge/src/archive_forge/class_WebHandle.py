from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class WebHandle(object):

    def __init__(self, url):
        self.url = url
        self.thumbprint = None
        self.ssl_context = None
        self.parsed_url = self._parse_url(url)
        self.https = self.parsed_url.group('scheme') == 'https://'
        if self.https:
            self.ssl_context = ssl._create_default_https_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
            self.thumbprint = self._get_thumbprint(self.parsed_url.group('hostname'))
            r = urlopen(url=url, context=self.ssl_context)
        else:
            r = urlopen(url)
        if r.code != 200:
            raise FileNotFoundError(url)
        self.headers = self._headers_to_dict(r)
        if 'accept-ranges' not in self.headers:
            raise Exception('Site does not accept ranges')
        self.st_size = int(self.headers['content-length'])
        self.offset = 0

    def _parse_url(self, url):
        HTTP_SCHEMA_PATTERN = "(?P<url>(?:(?P<scheme>[a-zA-Z]+:\\/\\/)?(?P<hostname>(?:[-a-zA-Z0-9@%_\\+~#=]{1,256}\\.){1,256}(?:[-a-zA-Z0-9@%_\\+~#=]{1,256})))(?::(?P<port>[[:digit:]]+))?(?P<path>(?:\\/[-a-zA-Z0-9!$&'()*+,\\\\\\/:;=@\\[\\]._~%]*)*)(?P<query>(?:(?:\\#|\\?)[-a-zA-Z0-9!$&'()*+,\\\\\\/:;=@\\[\\]._~]*)*))"
        return re.match(HTTP_SCHEMA_PATTERN, url)

    def _get_thumbprint(self, hostname):
        pem = ssl.get_server_certificate((hostname, 443))
        sha1 = hashlib.sha1(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()
        colon_notion = ':'.join((sha1[i:i + 2] for i in range(0, len(sha1), 2)))
        return None if sha1 is None else colon_notion

    def _headers_to_dict(self, r):
        result = {}
        if hasattr(r, 'getheaders'):
            for n, v in r.getheaders():
                result[n.lower()] = v.strip()
        else:
            for line in r.info().headers:
                if line.find(':') != -1:
                    n, v = line.split(': ', 1)
                    result[n.lower()] = v.strip()
        return result

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = self.st_size - offset
        return self.offset

    def seekable(self):
        return True

    def read(self, amount):
        start = self.offset
        end = self.offset + amount - 1
        req = Request(self.url, headers={'Range': 'bytes=%d-%d' % (start, end)})
        r = urlopen(req) if not self.ssl_context else urlopen(req, context=self.ssl_context)
        self.offset += amount
        result = r.read(amount)
        r.close()
        return result

    def progress(self):
        return int(100.0 * self.offset / self.st_size)