import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
class KubernetesTLSAuthConnection(KeyCertificateConnection):
    responseCls = KubernetesResponse
    timeout = 60

    def __init__(self, key, secure=True, host='localhost', port='6443', key_file=None, cert_file=None, **kwargs):
        super().__init__(key_file=key_file, cert_file=cert_file, secure=secure, host=host, port=port, url=None, proxy_url=None, timeout=None, backoff=None, retry_delay=None)
        if key_file:
            keypath = os.path.expanduser(key_file)
            is_file_path = os.path.exists(keypath) and os.path.isfile(keypath)
            if not is_file_path:
                raise InvalidCredsError('You need an key PEM file to authenticate via tls. For more info please visit:https://kubernetes.io/docs/concepts/cluster-administration/certificates/')
            self.key_file = key_file
            certpath = os.path.expanduser(cert_file)
            is_file_path = os.path.exists(certpath) and os.path.isfile(certpath)
            if not is_file_path:
                raise InvalidCredsError('You need an certificate PEM file to authenticate via tls. For more info please visit:https://kubernetes.io/docs/concepts/cluster-administration/certificates/')
            self.cert_file = cert_file

    def add_default_headers(self, headers):
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        return headers