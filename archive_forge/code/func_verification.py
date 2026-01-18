import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
@property
def verification(self):
    """
        The option for SSL verification given to underlying requests
        """
    return self.ca_cert if self.ca_cert is not None else self.verify