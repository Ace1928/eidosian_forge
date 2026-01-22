import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def _openid_args(self, callback_uri: str, ax_attrs: Iterable[str]=[], oauth_scope: Optional[str]=None) -> Dict[str, str]:
    handler = cast(RequestHandler, self)
    url = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
    args = {'openid.ns': 'http://specs.openid.net/auth/2.0', 'openid.claimed_id': 'http://specs.openid.net/auth/2.0/identifier_select', 'openid.identity': 'http://specs.openid.net/auth/2.0/identifier_select', 'openid.return_to': url, 'openid.realm': urllib.parse.urljoin(url, '/'), 'openid.mode': 'checkid_setup'}
    if ax_attrs:
        args.update({'openid.ns.ax': 'http://openid.net/srv/ax/1.0', 'openid.ax.mode': 'fetch_request'})
        ax_attrs = set(ax_attrs)
        required = []
        if 'name' in ax_attrs:
            ax_attrs -= set(['name', 'firstname', 'fullname', 'lastname'])
            required += ['firstname', 'fullname', 'lastname']
            args.update({'openid.ax.type.firstname': 'http://axschema.org/namePerson/first', 'openid.ax.type.fullname': 'http://axschema.org/namePerson', 'openid.ax.type.lastname': 'http://axschema.org/namePerson/last'})
        known_attrs = {'email': 'http://axschema.org/contact/email', 'language': 'http://axschema.org/pref/language', 'username': 'http://axschema.org/namePerson/friendly'}
        for name in ax_attrs:
            args['openid.ax.type.' + name] = known_attrs[name]
            required.append(name)
        args['openid.ax.required'] = ','.join(required)
    if oauth_scope:
        args.update({'openid.ns.oauth': 'http://specs.openid.net/extensions/oauth/1.0', 'openid.oauth.consumer': handler.request.host.split(':')[0], 'openid.oauth.scope': oauth_scope})
    return args