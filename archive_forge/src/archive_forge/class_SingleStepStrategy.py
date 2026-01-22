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
class SingleStepStrategy(object):

    @classmethod
    def supports(cls, path):
        return len(path) == 1

    def __init__(self, path):
        self.path = path

    def test(self, ignore_context):
        steps = self.path
        if steps[0][0] is ATTRIBUTE:
            steps = [_DOTSLASH] + steps
        select_attr = steps[-1][0] is ATTRIBUTE and steps[-1][1] or None
        counters = []
        depth = [0]

        def _test(event, namespaces, variables, updateonly=False):
            kind, data, pos = event[:3]
            if kind is END:
                if not ignore_context:
                    depth[0] -= 1
                return None
            elif kind is START_NS or kind is END_NS or kind is START_CDATA or (kind is END_CDATA):
                return None
            if not ignore_context:
                outside = steps[0][0] is SELF and depth[0] != 0 or (steps[0][0] is CHILD and depth[0] != 1) or (steps[0][0] is DESCENDANT and depth[0] < 1)
                if kind is START:
                    depth[0] += 1
                if outside:
                    return None
            axis, nodetest, predicates = steps[0]
            if not nodetest(kind, data, pos, namespaces, variables):
                return None
            if predicates:
                cnum = 0
                for predicate in predicates:
                    pretval = predicate(kind, data, pos, namespaces, variables)
                    if type(pretval) is float:
                        if len(counters) < cnum + 1:
                            counters.append(0)
                        counters[cnum] += 1
                        if counters[cnum] != int(pretval):
                            pretval = False
                        cnum += 1
                    if not pretval:
                        return None
            if select_attr:
                return select_attr(kind, data, pos, namespaces, variables)
            return True
        return _test