from __future__ import generator_stop
from ..util import FeedParserDict
def _start_gml_linearring(self, attrs_d):
    self.ingeometry = 'polygon'
    self.push('geometry', 0)