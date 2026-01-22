import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _get_cookie_csrf(self, proxy, timeout):
    if self._cookie is not None:
        utils.get_yf_logger().debug('reusing cookie')
        return True
    elif self._load_session_cookies():
        utils.get_yf_logger().debug('reusing persistent cookie')
        self._cookie = True
        return True
    base_args = {'headers': self.user_agent_headers, 'proxies': proxy, 'timeout': timeout}
    get_args = {**base_args, 'url': 'https://guce.yahoo.com/consent'}
    if self._session_is_caching:
        get_args['expire_after'] = self._expire_after
        response = self._session.get(**get_args)
    else:
        response = self._session.get(**get_args)
    soup = BeautifulSoup(response.content, 'html.parser')
    csrfTokenInput = soup.find('input', attrs={'name': 'csrfToken'})
    if csrfTokenInput is None:
        utils.get_yf_logger().debug('Failed to find "csrfToken" in response')
        return False
    csrfToken = csrfTokenInput['value']
    utils.get_yf_logger().debug(f'csrfToken = {csrfToken}')
    sessionIdInput = soup.find('input', attrs={'name': 'sessionId'})
    sessionId = sessionIdInput['value']
    utils.get_yf_logger().debug(f"sessionId='{sessionId}")
    originalDoneUrl = 'https://finance.yahoo.com/'
    namespace = 'yahoo'
    data = {'agree': ['agree', 'agree'], 'consentUUID': 'default', 'sessionId': sessionId, 'csrfToken': csrfToken, 'originalDoneUrl': originalDoneUrl, 'namespace': namespace}
    post_args = {**base_args, 'url': f'https://consent.yahoo.com/v2/collectConsent?sessionId={sessionId}', 'data': data}
    get_args = {**base_args, 'url': f'https://guce.yahoo.com/copyConsent?sessionId={sessionId}', 'data': data}
    if self._session_is_caching:
        post_args['expire_after'] = self._expire_after
        get_args['expire_after'] = self._expire_after
        self._session.post(**post_args)
        self._session.get(**get_args)
    else:
        self._session.post(**post_args)
        self._session.get(**get_args)
    self._cookie = True
    self._save_session_cookies()
    return True