import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
class SignedHTTPSAdapter(HTTPAdapter):

    def __init__(self, cert_file, key_file):
        self.cert_file = cert_file
        self.key_file = key_file
        super().__init__()

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, cert_file=self.cert_file, key_file=self.key_file)