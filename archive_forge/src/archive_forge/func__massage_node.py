from functools import reduce
from datetime import datetime
import re
def _massage_node(node, attr):
    """The real work for remove_rel is done here, parametrized with @rel and @rev"""
    if node.hasAttribute('property') and node.hasAttribute(attr):
        vals = node.getAttribute(attr).strip().split()
        if len(vals) != 0:
            final_vals = [v for v in vals if not termname.match(v)]
            if len(final_vals) == 0:
                node.removeAttribute(attr)
            else:
                node.setAttribute(attr, reduce(lambda x, y: x + ' ' + y, final_vals))