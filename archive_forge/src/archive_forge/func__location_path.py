from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
def _location_path(self):
    steps = []
    while True:
        if self.cur_token.startswith('/'):
            if not steps:
                if self.cur_token == '//':
                    self.next_token()
                    axis, nodetest, predicates = self._location_step()
                    steps.append((DESCENDANT_OR_SELF, nodetest, predicates))
                    if self.at_end or not self.cur_token.startswith('/'):
                        break
                    continue
                else:
                    raise PathSyntaxError('Absolute location paths not supported', self.filename, self.lineno)
            elif self.cur_token == '//':
                steps.append((DESCENDANT_OR_SELF, NodeTest(), []))
            self.next_token()
        axis, nodetest, predicates = self._location_step()
        if not axis:
            axis = CHILD
        steps.append((axis, nodetest, predicates))
        if self.at_end or not self.cur_token.startswith('/'):
            break
    return steps