import re
def _calcPlaces(self, V):
    """called with the full set of values to be formatted so we can calculate places"""
    self.places = max([len(_tz_re.sub('', _ld_re.sub('', str(v)))) for v in V])