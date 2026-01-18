import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def vars2fortran(block, vars, args, tab='', as_interface=False):
    setmesstext(block)
    ret = ''
    nout = []
    for a in args:
        if a in block['vars']:
            nout.append(a)
    if 'commonvars' in block:
        for a in block['commonvars']:
            if a in vars:
                if a not in nout:
                    nout.append(a)
            else:
                errmess('vars2fortran: Confused?!: "%s" is not defined in vars.\n' % a)
    if 'varnames' in block:
        nout.extend(block['varnames'])
    if not as_interface:
        for a in list(vars.keys()):
            if a not in nout:
                nout.append(a)
    for a in nout:
        if 'depend' in vars[a]:
            for d in vars[a]['depend']:
                if d in vars and 'depend' in vars[d] and (a in vars[d]['depend']):
                    errmess('vars2fortran: Warning: cross-dependence between variables "%s" and "%s"\n' % (a, d))
        if 'externals' in block and a in block['externals']:
            if isintent_callback(vars[a]):
                ret = '%s%sintent(callback) %s' % (ret, tab, a)
            ret = '%s%sexternal %s' % (ret, tab, a)
            if isoptional(vars[a]):
                ret = '%s%soptional %s' % (ret, tab, a)
            if a in vars and 'typespec' not in vars[a]:
                continue
            cont = 1
            for b in block['body']:
                if a == b['name'] and b['block'] == 'function':
                    cont = 0
                    break
            if cont:
                continue
        if a not in vars:
            show(vars)
            outmess('vars2fortran: No definition for argument "%s".\n' % a)
            continue
        if a == block['name']:
            if block['block'] != 'function' or block.get('result'):
                continue
        if 'typespec' not in vars[a]:
            if 'attrspec' in vars[a] and 'external' in vars[a]['attrspec']:
                if a in args:
                    ret = '%s%sexternal %s' % (ret, tab, a)
                continue
            show(vars[a])
            outmess('vars2fortran: No typespec for argument "%s".\n' % a)
            continue
        vardef = vars[a]['typespec']
        if vardef == 'type' and 'typename' in vars[a]:
            vardef = '%s(%s)' % (vardef, vars[a]['typename'])
        selector = {}
        if 'kindselector' in vars[a]:
            selector = vars[a]['kindselector']
        elif 'charselector' in vars[a]:
            selector = vars[a]['charselector']
        if '*' in selector:
            if selector['*'] in ['*', ':']:
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
        c = ' '
        if 'attrspec' in vars[a]:
            attr = [l for l in vars[a]['attrspec'] if l not in ['external']]
            if as_interface and 'intent(in)' in attr and ('intent(out)' in attr):
                attr.remove('intent(out)')
            if attr:
                vardef = '%s, %s' % (vardef, ','.join(attr))
                c = ','
        if 'dimension' in vars[a]:
            vardef = '%s%sdimension(%s)' % (vardef, c, ','.join(vars[a]['dimension']))
            c = ','
        if 'intent' in vars[a]:
            lst = true_intent_list(vars[a])
            if lst:
                vardef = '%s%sintent(%s)' % (vardef, c, ','.join(lst))
            c = ','
        if 'check' in vars[a]:
            vardef = '%s%scheck(%s)' % (vardef, c, ','.join(vars[a]['check']))
            c = ','
        if 'depend' in vars[a]:
            vardef = '%s%sdepend(%s)' % (vardef, c, ','.join(vars[a]['depend']))
            c = ','
        if '=' in vars[a]:
            v = vars[a]['=']
            if vars[a]['typespec'] in ['complex', 'double complex']:
                try:
                    v = eval(v)
                    v = '(%s,%s)' % (v.real, v.imag)
                except Exception:
                    pass
            vardef = '%s :: %s=%s' % (vardef, a, v)
        else:
            vardef = '%s :: %s' % (vardef, a)
        ret = '%s%s%s' % (ret, tab, vardef)
    return ret