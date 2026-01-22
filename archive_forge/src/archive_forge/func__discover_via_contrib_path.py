import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def _discover_via_contrib_path(version):
    module_path = os.path.dirname(os.path.abspath(__file__))
    version_str = 'v%s' % version.replace('.', '_')
    ext_path = os.path.join(module_path, version_str, 'contrib')
    ext_glob = os.path.join(ext_path, '*.py')
    for ext_path in glob.iglob(ext_glob):
        name = os.path.basename(ext_path)[:-3]
        if name == '__init__':
            continue
        module = load_module(name, ext_path)
        yield (name, module)