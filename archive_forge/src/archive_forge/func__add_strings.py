import os
from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none
def _add_strings(context, nodes, add_slash=False):
    string = ''
    first = True
    for child_node in nodes:
        values = context.infer_node(child_node)
        if len(values) != 1:
            return None
        c, = values
        s = get_str_or_none(c)
        if s is None:
            return None
        if not first and add_slash:
            string += os.path.sep
        string += s
        first = False
    return string