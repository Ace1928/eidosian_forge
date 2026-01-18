from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def unsuppress(self, lineno, idlist):
    for idstr in idlist:
        if idstr not in self._suppressions:
            logger.warning('Unsupressing %s which is not currently surpressed', idstr)
        self._suppressions.discard(idstr)
    self._suppression_events.append(SuppressionEvent(lineno, 'remove', list(idlist)))