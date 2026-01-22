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
def _node_type(self):
    name = self.cur_token
    self.next_token()
    args = []
    if self.cur_token != '()':
        self.next_token()
        if self.cur_token != ')':
            string = self.cur_token
            if (string[0], string[-1]) in self._QUOTES:
                string = string[1:-1]
            args.append(string)
    cls = _nodetest_map.get(name)
    if not cls:
        raise PathSyntaxError('%s() not allowed here' % name, self.filename, self.lineno)
    return cls(*args)