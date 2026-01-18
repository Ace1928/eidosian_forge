from .util import subvals
def wrap_nary_f(fun, op, argnum):
    namestr = '{op}_of_{fun}_wrt_argnum_{argnum}'
    docstr = '    {op} of function {fun} with respect to argument number {argnum}. Takes the\n    same arguments as {fun} but returns the {op}.\n    '
    return wraps(fun, namestr, docstr, op=get_name(op), argnum=argnum)