import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _make_help_call(target: str, esc: str, lspace: str) -> str:
    """Prepares a pinfo(2)/psearch call from a target name and the escape
    (i.e. ? or ??)"""
    method = 'pinfo2' if esc == '??' else 'psearch' if '*' in target else 'pinfo'
    arg = ' '.join([method, target])
    t_magic_name, _, t_magic_arg_s = arg.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return '%sget_ipython().run_line_magic(%r, %r)' % (lspace, t_magic_name, t_magic_arg_s)