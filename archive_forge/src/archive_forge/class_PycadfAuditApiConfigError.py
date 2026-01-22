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
class PycadfAuditApiConfigError(Exception):
    """Error raised when pyCADF fails to configure correctly."""
    pass