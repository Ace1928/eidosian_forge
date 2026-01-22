import collections
import configparser
import re
from oslo_log import log as logging
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import eventfactory as factory
from pycadf import host
from pycadf import identifier
from pycadf import resource
from pycadf import tag
from urllib import parse as urlparse
def _build_typeURI(self, req, service_type):
    """Build typeURI of target.

        Combines service type and corresponding path for greater detail.
        """
    type_uri = ''
    prev_key = None
    for key in re.split('/', req.path):
        key = self._clean_path(key)
        if key in self._MAP.path_kw:
            type_uri += '/' + key
        elif prev_key in self._MAP.path_kw:
            type_uri += '/' + self._MAP.path_kw[prev_key]
        prev_key = key
    return service_type + type_uri