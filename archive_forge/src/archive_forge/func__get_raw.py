import importlib
import logging
import os
import sys
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import session as ks_session
from oslo_utils import importutils
from barbicanclient import exceptions
def _get_raw(self, path, *args, **kwargs):
    return self.request(path, 'GET', *args, **kwargs).content