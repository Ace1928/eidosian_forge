import copy
from .auxfuncs import (
from ._isocbind import isoc_kindmap
def var2fixfortran(vars, a, fa=None, f90mode=None):
    if fa is None:
        fa = a
    if a not in vars:
        show(vars)
        outmess('var2fixfortran: No definition for argument "%s".\n' % a)
        return ''
    if 'typespec' not in vars[a]:
        show(vars[a])
        outmess('var2fixfortran: No typespec for argument "%s".\n' % a)
        return ''
    vardef = vars[a]['typespec']
    if vardef == 'type' and 'typename' in vars[a]:
        vardef = '%s(%s)' % (vardef, vars[a]['typename'])
    selector = {}
    lk = ''
    if 'kindselector' in vars[a]:
        selector = vars[a]['kindselector']
        lk = 'kind'
    elif 'charselector' in vars[a]:
        selector = vars[a]['charselector']
        lk = 'len'
    if '*' in selector:
        if f90mode:
            if selector['*'] in ['*', ':', '(*)']:
                vardef = '%s(len=*)' % vardef
            else:
                vardef = '%s(%s=%s)' % (vardef, lk, selector['*'])
        elif selector['*'] in ['*', ':']:
            vardef = '%s*(%s)' % (vardef, selector['*'])
        else:
            vardef = '%s*%s' % (vardef, selector['*'])
    elif 'len' in selector:
        vardef = '%s(len=%s' % (vardef, selector['len'])
        if 'kind' in selector:
            vardef = '%s,kind=%s)' % (vardef, selector['kind'])
        else:
            vardef = '%s)' % vardef
    elif 'kind' in selector:
        vardef = '%s(kind=%s)' % (vardef, selector['kind'])
    vardef = '%s %s' % (vardef, fa)
    if 'dimension' in vars[a]:
        vardef = '%s(%s)' % (vardef, ','.join(vars[a]['dimension']))
    return vardef