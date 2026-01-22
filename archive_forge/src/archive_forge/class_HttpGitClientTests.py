import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class HttpGitClientTests(TestCase):

    def test_get_url(self):
        base_url = 'https://github.com/jelmer/dulwich'
        path = '/jelmer/dulwich'
        c = HttpGitClient(base_url)
        url = c.get_url(path)
        self.assertEqual('https://github.com/jelmer/dulwich', url)

    def test_get_url_bytes_path(self):
        base_url = 'https://github.com/jelmer/dulwich'
        path_bytes = b'/jelmer/dulwich'
        c = HttpGitClient(base_url)
        url = c.get_url(path_bytes)
        self.assertEqual('https://github.com/jelmer/dulwich', url)

    def test_get_url_with_username_and_passwd(self):
        base_url = 'https://github.com/jelmer/dulwich'
        path = '/jelmer/dulwich'
        c = HttpGitClient(base_url, username='USERNAME', password='PASSWD')
        url = c.get_url(path)
        self.assertEqual('https://github.com/jelmer/dulwich', url)

    def test_init_username_passwd_set(self):
        url = 'https://github.com/jelmer/dulwich'
        c = HttpGitClient(url, config=None, username='user', password='passwd')
        self.assertEqual('user', c._username)
        self.assertEqual('passwd', c._password)
        basic_auth = c.pool_manager.headers['authorization']
        auth_string = '{}:{}'.format('user', 'passwd')
        b64_credentials = base64.b64encode(auth_string.encode('latin1'))
        expected_basic_auth = 'Basic %s' % b64_credentials.decode('latin1')
        self.assertEqual(basic_auth, expected_basic_auth)

    def test_init_username_set_no_password(self):
        url = 'https://github.com/jelmer/dulwich'
        c = HttpGitClient(url, config=None, username='user')
        self.assertEqual('user', c._username)
        self.assertIsNone(c._password)
        basic_auth = c.pool_manager.headers['authorization']
        auth_string = b'user:'
        b64_credentials = base64.b64encode(auth_string)
        expected_basic_auth = f'Basic {b64_credentials.decode('ascii')}'
        self.assertEqual(basic_auth, expected_basic_auth)

    def test_init_no_username_passwd(self):
        url = 'https://github.com/jelmer/dulwich'
        c = HttpGitClient(url, config=None)
        self.assertIs(None, c._username)
        self.assertIs(None, c._password)
        self.assertNotIn('authorization', c.pool_manager.headers)

    def test_from_parsedurl_username_only(self):
        username = 'user'
        url = f'https://{username}@github.com/jelmer/dulwich'
        c = HttpGitClient.from_parsedurl(urlparse(url))
        self.assertEqual(c._username, username)
        self.assertEqual(c._password, None)
        basic_auth = c.pool_manager.headers['authorization']
        auth_string = username.encode('ascii') + b':'
        b64_credentials = base64.b64encode(auth_string)
        expected_basic_auth = f'Basic {b64_credentials.decode('ascii')}'
        self.assertEqual(basic_auth, expected_basic_auth)

    def test_from_parsedurl_on_url_with_quoted_credentials(self):
        original_username = 'john|the|first'
        quoted_username = urlquote(original_username)
        original_password = 'Ya#1$2%3'
        quoted_password = urlquote(original_password)
        url = f'https://{quoted_username}:{quoted_password}@github.com/jelmer/dulwich'
        c = HttpGitClient.from_parsedurl(urlparse(url))
        self.assertEqual(original_username, c._username)
        self.assertEqual(original_password, c._password)
        basic_auth = c.pool_manager.headers['authorization']
        auth_string = f'{original_username}:{original_password}'
        b64_credentials = base64.b64encode(auth_string.encode('latin1'))
        expected_basic_auth = 'Basic %s' % b64_credentials.decode('latin1')
        self.assertEqual(basic_auth, expected_basic_auth)

    def test_url_redirect_location(self):
        from urllib3.response import HTTPResponse
        test_data = {'https://gitlab.com/inkscape/inkscape/': {'location': 'https://gitlab.com/inkscape/inkscape.git/', 'redirect_url': 'https://gitlab.com/inkscape/inkscape.git/', 'refs_data': b'001e# service=git-upload-pack\n00000032fb2bebf4919a011f0fd7cec085443d0031228e76 HEAD\n0000'}, 'https://github.com/jelmer/dulwich/': {'location': 'https://github.com/jelmer/dulwich/', 'redirect_url': 'https://github.com/jelmer/dulwich/', 'refs_data': b'001e# service=git-upload-pack\n000000323ff25e09724aa4d86ea5bca7d5dd0399a3c8bfcf HEAD\n0000'}, 'https://codeberg.org/ashwinvis/radicale-sh.git/': {'location': '/ashwinvis/radicale-auth-sh/', 'redirect_url': 'https://codeberg.org/ashwinvis/radicale-auth-sh/', 'refs_data': b'001e# service=git-upload-pack\n00000032470f8603768b608fc988675de2fae8f963c21158 HEAD\n0000'}}
        tail = 'info/refs?service=git-upload-pack'

        class PoolManagerMock:

            def __init__(self) -> None:
                self.headers: Dict[str, str] = {}

            def request(self, method, url, fields=None, headers=None, redirect=True, preload_content=True):
                base_url = url[:-len(tail)]
                redirect_base_url = test_data[base_url]['location']
                redirect_url = redirect_base_url + tail
                headers = {'Content-Type': 'application/x-git-upload-pack-advertisement'}
                body = test_data[base_url]['refs_data']
                status = 200
                request_url = redirect_url
                if redirect is False:
                    request_url = url
                    if redirect_base_url != base_url:
                        body = b''
                        headers['location'] = test_data[base_url]['location']
                        status = 301
                return HTTPResponse(body=BytesIO(body), headers=headers, request_method=method, request_url=request_url, preload_content=preload_content, status=status)
        pool_manager = PoolManagerMock()
        for base_url in test_data.keys():
            c = HttpGitClient(base_url, pool_manager=pool_manager, config=None)
            _, _, processed_url = c._discover_references(b'git-upload-pack', base_url)
            resp = c.pool_manager.request('GET', base_url + tail, redirect=False)
            redirect_location = resp.get_redirect_location()
            if resp.status == 200:
                self.assertFalse(redirect_location)
            if redirect_location:
                self.assertEqual(processed_url, test_data[base_url]['redirect_url'])
            else:
                self.assertEqual(processed_url, base_url)

    def test_smart_request_content_type_with_directive_check(self):
        from urllib3.response import HTTPResponse

        class PoolManagerMock:

            def __init__(self) -> None:
                self.headers: Dict[str, str] = {}

            def request(self, method, url, fields=None, headers=None, redirect=True, preload_content=True):
                return HTTPResponse(headers={'Content-Type': 'application/x-git-upload-pack-result; charset=utf-8'}, request_method=method, request_url=url, preload_content=preload_content, status=200)
        clone_url = 'https://hacktivis.me/git/blog.git/'
        client = HttpGitClient(clone_url, pool_manager=PoolManagerMock(), config=None)
        self.assertTrue(client._smart_request('git-upload-pack', clone_url, data=None))