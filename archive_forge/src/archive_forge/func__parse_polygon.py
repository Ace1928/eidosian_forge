import re
def _parse_polygon(self, poly):
    if 'points' in poly.attrib:
        self._start_path('M' + poly.attrib['points'])
        self._end_path()