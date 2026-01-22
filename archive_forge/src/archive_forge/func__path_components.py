import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
def _path_components(self):
    """Return a list of the path components."""
    return self.path.lstrip('/').split('/')