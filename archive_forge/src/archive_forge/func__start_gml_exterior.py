from __future__ import generator_stop
from ..util import FeedParserDict
def _start_gml_exterior(self, attrs_d):
    self.push('geometry', 0)