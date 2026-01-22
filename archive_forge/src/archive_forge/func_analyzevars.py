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
def analyzevars(block):
    """
    Sets correct dimension information for each variable/parameter
    """
    global f90modulevars
    setmesstext(block)
    implicitrules, attrrules = buildimplicitrules(block)
    vars = copy.copy(block['vars'])
    if block['block'] == 'function' and block['name'] not in vars:
        vars[block['name']] = {}
    if '' in block['vars']:
        del vars['']
        if 'attrspec' in block['vars']['']:
            gen = block['vars']['']['attrspec']
            for n in set(vars) | set((b['name'] for b in block['body'])):
                for k in ['public', 'private']:
                    if k in gen:
                        vars[n] = setattrspec(vars.get(n, {}), k)
    svars = []
    args = block['args']
    for a in args:
        try:
            vars[a]
            svars.append(a)
        except KeyError:
            pass
    for n in list(vars.keys()):
        if n not in args:
            svars.append(n)
    params = get_parameters(vars, get_useparameters(block))
    dep_matches = {}
    name_match = re.compile('[A-Za-z][\\w$]*').match
    for v in list(vars.keys()):
        m = name_match(v)
        if m:
            n = v[m.start():m.end()]
            try:
                dep_matches[n]
            except KeyError:
                dep_matches[n] = re.compile('.*\\b%s\\b' % v, re.I).match
    for n in svars:
        if n[0] in list(attrrules.keys()):
            vars[n] = setattrspec(vars[n], attrrules[n[0]])
        if 'typespec' not in vars[n]:
            if not ('attrspec' in vars[n] and 'external' in vars[n]['attrspec']):
                if implicitrules:
                    ln0 = n[0].lower()
                    for k in list(implicitrules[ln0].keys()):
                        if k == 'typespec' and implicitrules[ln0][k] == 'undefined':
                            continue
                        if k not in vars[n]:
                            vars[n][k] = implicitrules[ln0][k]
                        elif k == 'attrspec':
                            for l in implicitrules[ln0][k]:
                                vars[n] = setattrspec(vars[n], l)
                elif n in block['args']:
                    outmess('analyzevars: typespec of variable %s is not defined in routine %s.\n' % (repr(n), block['name']))
        if 'charselector' in vars[n]:
            if 'len' in vars[n]['charselector']:
                l = vars[n]['charselector']['len']
                try:
                    l = str(eval(l, {}, params))
                except Exception:
                    pass
                vars[n]['charselector']['len'] = l
        if 'kindselector' in vars[n]:
            if 'kind' in vars[n]['kindselector']:
                l = vars[n]['kindselector']['kind']
                try:
                    l = str(eval(l, {}, params))
                except Exception:
                    pass
                vars[n]['kindselector']['kind'] = l
        dimension_exprs = {}
        if 'attrspec' in vars[n]:
            attr = vars[n]['attrspec']
            attr.reverse()
            vars[n]['attrspec'] = []
            dim, intent, depend, check, note = (None, None, None, None, None)
            for a in attr:
                if a[:9] == 'dimension':
                    dim = a[9:].strip()[1:-1]
                elif a[:6] == 'intent':
                    intent = a[6:].strip()[1:-1]
                elif a[:6] == 'depend':
                    depend = a[6:].strip()[1:-1]
                elif a[:5] == 'check':
                    check = a[5:].strip()[1:-1]
                elif a[:4] == 'note':
                    note = a[4:].strip()[1:-1]
                else:
                    vars[n] = setattrspec(vars[n], a)
                if intent:
                    if 'intent' not in vars[n]:
                        vars[n]['intent'] = []
                    for c in [x.strip() for x in markoutercomma(intent).split('@,@')]:
                        tmp = c.replace(' ', '')
                        if tmp not in vars[n]['intent']:
                            vars[n]['intent'].append(tmp)
                    intent = None
                if note:
                    note = note.replace('\\n\\n', '\n\n')
                    note = note.replace('\\n ', '\n')
                    if 'note' not in vars[n]:
                        vars[n]['note'] = [note]
                    else:
                        vars[n]['note'].append(note)
                    note = None
                if depend is not None:
                    if 'depend' not in vars[n]:
                        vars[n]['depend'] = []
                    for c in rmbadname([x.strip() for x in markoutercomma(depend).split('@,@')]):
                        if c not in vars[n]['depend']:
                            vars[n]['depend'].append(c)
                    depend = None
                if check is not None:
                    if 'check' not in vars[n]:
                        vars[n]['check'] = []
                    for c in [x.strip() for x in markoutercomma(check).split('@,@')]:
                        if c not in vars[n]['check']:
                            vars[n]['check'].append(c)
                    check = None
            if dim and 'dimension' not in vars[n]:
                vars[n]['dimension'] = []
                for d in rmbadname([x.strip() for x in markoutercomma(dim).split('@,@')]):
                    try:
                        d = param_parse(d, params)
                    except (ValueError, IndexError, KeyError):
                        outmess(f'analyzevars: could not parse dimension for variable {d!r}\n')
                    dim_char = ':' if d == ':' else '*'
                    if d == dim_char:
                        dl = [dim_char]
                    else:
                        dl = markoutercomma(d, ':').split('@:@')
                    if len(dl) == 2 and '*' in dl:
                        dl = ['*']
                        d = '*'
                    if len(dl) == 1 and dl[0] != dim_char:
                        dl = ['1', dl[0]]
                    if len(dl) == 2:
                        d1, d2 = map(symbolic.Expr.parse, dl)
                        dsize = d2 - d1 + 1
                        d = dsize.tostring(language=symbolic.Language.C)
                        solver_and_deps = {}
                        for v in block['vars']:
                            s = symbolic.as_symbol(v)
                            if dsize.contains(s):
                                try:
                                    a, b = dsize.linear_solve(s)

                                    def solve_v(s, a=a, b=b):
                                        return (s - b) / a
                                    all_symbols = set(a.symbols())
                                    all_symbols.update(b.symbols())
                                except RuntimeError as msg:
                                    solve_v = None
                                    all_symbols = set(dsize.symbols())
                                v_deps = set((s.data for s in all_symbols if s.data in vars))
                                solver_and_deps[v] = (solve_v, list(v_deps))
                        dimension_exprs[d] = solver_and_deps
                    vars[n]['dimension'].append(d)
        if 'check' not in vars[n] and 'args' in block and (n in block['args']):
            n_deps = vars[n].get('depend', [])
            n_checks = []
            n_is_input = l_or(isintent_in, isintent_inout, isintent_inplace)(vars[n])
            if isarray(vars[n]):
                for i, d in enumerate(vars[n]['dimension']):
                    coeffs_and_deps = dimension_exprs.get(d)
                    if coeffs_and_deps is None:
                        pass
                    elif n_is_input:
                        for v, (solver, deps) in coeffs_and_deps.items():

                            def compute_deps(v, deps):
                                for v1 in coeffs_and_deps.get(v, [None, []])[1]:
                                    if v1 not in deps:
                                        deps.add(v1)
                                        compute_deps(v1, deps)
                            all_deps = set()
                            compute_deps(v, all_deps)
                            if v in n_deps or '=' in vars[v] or 'depend' in vars[v]:
                                continue
                            if solver is not None and v not in all_deps:
                                is_required = False
                                init = solver(symbolic.as_symbol(f'shape({n}, {i})'))
                                init = init.tostring(language=symbolic.Language.C)
                                vars[v]['='] = init
                                vars[v]['depend'] = [n] + deps
                                if 'check' not in vars[v]:
                                    vars[v]['check'] = [f'shape({n}, {i}) == {d}']
                            else:
                                is_required = True
                                if 'intent' not in vars[v]:
                                    vars[v]['intent'] = []
                                if 'in' not in vars[v]['intent']:
                                    vars[v]['intent'].append('in')
                                n_deps.append(v)
                                n_checks.append(f'shape({n}, {i}) == {d}')
                            v_attr = vars[v].get('attrspec', [])
                            if not ('optional' in v_attr or 'required' in v_attr):
                                v_attr.append('required' if is_required else 'optional')
                            if v_attr:
                                vars[v]['attrspec'] = v_attr
                    if coeffs_and_deps is not None:
                        for v, (solver, deps) in coeffs_and_deps.items():
                            v_deps = vars[v].get('depend', [])
                            for aa in vars[v].get('attrspec', []):
                                if aa.startswith('depend'):
                                    aa = ''.join(aa.split())
                                    v_deps.extend(aa[7:-1].split(','))
                            if v_deps:
                                vars[v]['depend'] = list(set(v_deps))
                            if n not in v_deps:
                                n_deps.append(v)
            elif isstring(vars[n]):
                if 'charselector' in vars[n]:
                    if '*' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['*'], params)
                        vars[n]['charselector']['*'] = length
                    elif 'len' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['len'], params)
                        del vars[n]['charselector']['len']
                        vars[n]['charselector']['*'] = length
            if n_checks:
                vars[n]['check'] = n_checks
            if n_deps:
                vars[n]['depend'] = list(set(n_deps))
        if '=' in vars[n]:
            if 'attrspec' not in vars[n]:
                vars[n]['attrspec'] = []
            if 'optional' not in vars[n]['attrspec'] and 'required' not in vars[n]['attrspec']:
                vars[n]['attrspec'].append('optional')
            if 'depend' not in vars[n]:
                vars[n]['depend'] = []
                for v, m in list(dep_matches.items()):
                    if m(vars[n]['=']):
                        vars[n]['depend'].append(v)
                if not vars[n]['depend']:
                    del vars[n]['depend']
            if isscalar(vars[n]):
                vars[n]['='] = _eval_scalar(vars[n]['='], params)
    for n in list(vars.keys()):
        if n == block['name']:
            if 'note' in vars[n]:
                block['note'] = vars[n]['note']
            if block['block'] == 'function':
                if 'result' in block and block['result'] in vars:
                    vars[n] = appenddecl(vars[n], vars[block['result']])
                if 'prefix' in block:
                    pr = block['prefix']
                    pr1 = pr.replace('pure', '')
                    ispure = not pr == pr1
                    pr = pr1.replace('recursive', '')
                    isrec = not pr == pr1
                    m = typespattern[0].match(pr)
                    if m:
                        typespec, selector, attr, edecl = cracktypespec0(m.group('this'), m.group('after'))
                        kindselect, charselect, typename = cracktypespec(typespec, selector)
                        vars[n]['typespec'] = typespec
                        try:
                            if block['result']:
                                vars[block['result']]['typespec'] = typespec
                        except Exception:
                            pass
                        if kindselect:
                            if 'kind' in kindselect:
                                try:
                                    kindselect['kind'] = eval(kindselect['kind'], {}, params)
                                except Exception:
                                    pass
                            vars[n]['kindselector'] = kindselect
                        if charselect:
                            vars[n]['charselector'] = charselect
                        if typename:
                            vars[n]['typename'] = typename
                        if ispure:
                            vars[n] = setattrspec(vars[n], 'pure')
                        if isrec:
                            vars[n] = setattrspec(vars[n], 'recursive')
                    else:
                        outmess('analyzevars: prefix (%s) were not used\n' % repr(block['prefix']))
    if not block['block'] in ['module', 'pythonmodule', 'python module', 'block data']:
        if 'commonvars' in block:
            neededvars = copy.copy(block['args'] + block['commonvars'])
        else:
            neededvars = copy.copy(block['args'])
        for n in list(vars.keys()):
            if l_or(isintent_callback, isintent_aux)(vars[n]):
                neededvars.append(n)
        if 'entry' in block:
            neededvars.extend(list(block['entry'].keys()))
            for k in list(block['entry'].keys()):
                for n in block['entry'][k]:
                    if n not in neededvars:
                        neededvars.append(n)
        if block['block'] == 'function':
            if 'result' in block:
                neededvars.append(block['result'])
            else:
                neededvars.append(block['name'])
        if block['block'] in ['subroutine', 'function']:
            name = block['name']
            if name in vars and 'intent' in vars[name]:
                block['intent'] = vars[name]['intent']
        if block['block'] == 'type':
            neededvars.extend(list(vars.keys()))
        for n in list(vars.keys()):
            if n not in neededvars:
                del vars[n]
    return vars