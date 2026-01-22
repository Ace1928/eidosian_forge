import logging
from reportlab import rl_config
def _aSpaceString(self):
    return '(%s x %s%s)' % (self._getAvailableWidth(), self._aH, self._atTop and '*' or '')