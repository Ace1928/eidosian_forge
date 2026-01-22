import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_feed(self, attrs_d):
    self.infeed = 1
    versionmap = {'0.1': 'atom01', '0.2': 'atom02', '0.3': 'atom03'}
    if not self.version:
        attr_version = attrs_d.get('version')
        version = versionmap.get(attr_version)
        if version:
            self.version = version
        else:
            self.version = 'atom'