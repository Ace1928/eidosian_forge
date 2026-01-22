import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
def _list_repr_elided(v, threshold=200, edgeitems=3, indent=0, width=80):
    """
    Return a string representation for of a list where list is elided if
    it has more than n elements

    Parameters
    ----------
    v : list
        Input list
    threshold :
        Maximum number of elements to display

    Returns
    -------
    str
    """
    if isinstance(v, list):
        open_char, close_char = ('[', ']')
    elif isinstance(v, tuple):
        open_char, close_char = ('(', ')')
    else:
        raise ValueError('Invalid value of type: %s' % type(v))
    if len(v) <= threshold:
        disp_v = v
    else:
        disp_v = list(v[:edgeitems]) + ['...'] + list(v[-edgeitems:])
    v_str = open_char + ', '.join([str(e) for e in disp_v]) + close_char
    v_wrapped = '\n'.join(textwrap.wrap(v_str, width=width, initial_indent=' ' * (indent + 1), subsequent_indent=' ' * (indent + 1))).strip()
    return v_wrapped