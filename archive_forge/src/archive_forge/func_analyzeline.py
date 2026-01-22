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
def analyzeline(m, case, line):
    """
    Reads each line in the input file in sequence and updates global vars.

    Effectively reads and collects information from the input file to the
    global variable groupcache, a dictionary containing info about each part
    of the fortran module.

    At the end of analyzeline, information is filtered into the correct dict
    keys, but parameter values and dimensions are not yet interpreted.
    """
    global groupcounter, groupname, groupcache, grouplist, filepositiontext
    global currentfilename, f77modulename, neededinterface, neededmodule
    global expectbegin, gotnextfile, previous_context
    block = m.group('this')
    if case != 'multiline':
        previous_context = None
    if expectbegin and case not in ['begin', 'call', 'callfun', 'type'] and (not skipemptyends) and (groupcounter < 1):
        newname = os.path.basename(currentfilename).split('.')[0]
        outmess('analyzeline: no group yet. Creating program group with name "%s".\n' % newname)
        gotnextfile = 0
        groupcounter = groupcounter + 1
        groupname[groupcounter] = 'program'
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = 'program'
        groupcache[groupcounter]['name'] = newname
        groupcache[groupcounter]['from'] = 'fromsky'
        expectbegin = 0
    if case in ['begin', 'call', 'callfun']:
        block = block.lower()
        if re.match('block\\s*data', block, re.I):
            block = 'block data'
        elif re.match('python\\s*module', block, re.I):
            block = 'python module'
        elif re.match('abstract\\s*interface', block, re.I):
            block = 'abstract interface'
        if block == 'type':
            name, attrs, _ = _resolvetypedefpattern(m.group('after'))
            groupcache[groupcounter]['vars'][name] = dict(attrspec=attrs)
            args = []
            result = None
        else:
            name, args, result, bindcline = _resolvenameargspattern(m.group('after'))
        if name is None:
            if block == 'block data':
                name = '_BLOCK_DATA_'
            else:
                name = ''
            if block not in ['interface', 'block data', 'abstract interface']:
                outmess('analyzeline: No name/args pattern found for line.\n')
        previous_context = (block, name, groupcounter)
        if args:
            args = rmbadname([x.strip() for x in markoutercomma(args).split('@,@')])
        else:
            args = []
        if '' in args:
            while '' in args:
                args.remove('')
            outmess('analyzeline: argument list is malformed (missing argument).\n')
        needmodule = 0
        needinterface = 0
        if case in ['call', 'callfun']:
            needinterface = 1
            if 'args' not in groupcache[groupcounter]:
                return
            if name not in groupcache[groupcounter]['args']:
                return
            for it in grouplist[groupcounter]:
                if it['name'] == name:
                    return
            if name in groupcache[groupcounter]['interfaced']:
                return
            block = {'call': 'subroutine', 'callfun': 'function'}[case]
        if f77modulename and neededmodule == -1 and (groupcounter <= 1):
            neededmodule = groupcounter + 2
            needmodule = 1
            if block not in ['interface', 'abstract interface']:
                needinterface = 1
        groupcounter = groupcounter + 1
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        if needmodule:
            if verbose > 1:
                outmess('analyzeline: Creating module block %s\n' % repr(f77modulename), 0)
            groupname[groupcounter] = 'module'
            groupcache[groupcounter]['block'] = 'python module'
            groupcache[groupcounter]['name'] = f77modulename
            groupcache[groupcounter]['from'] = ''
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        if needinterface:
            if verbose > 1:
                outmess('analyzeline: Creating additional interface block (groupcounter=%s).\n' % groupcounter, 0)
            groupname[groupcounter] = 'interface'
            groupcache[groupcounter]['block'] = 'interface'
            groupcache[groupcounter]['name'] = 'unknown_interface'
            groupcache[groupcounter]['from'] = '%s:%s' % (groupcache[groupcounter - 1]['from'], groupcache[groupcounter - 1]['name'])
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        groupname[groupcounter] = block
        groupcache[groupcounter]['block'] = block
        if not name:
            name = 'unknown_' + block.replace(' ', '_')
        groupcache[groupcounter]['prefix'] = m.group('before')
        groupcache[groupcounter]['name'] = rmbadname1(name)
        groupcache[groupcounter]['result'] = result
        if groupcounter == 1:
            groupcache[groupcounter]['from'] = currentfilename
        elif f77modulename and groupcounter == 3:
            groupcache[groupcounter]['from'] = '%s:%s' % (groupcache[groupcounter - 1]['from'], currentfilename)
        else:
            groupcache[groupcounter]['from'] = '%s:%s' % (groupcache[groupcounter - 1]['from'], groupcache[groupcounter - 1]['name'])
        for k in list(groupcache[groupcounter].keys()):
            if not groupcache[groupcounter][k]:
                del groupcache[groupcounter][k]
        groupcache[groupcounter]['args'] = args
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['externals'] = []
        groupcache[groupcounter]['interfaced'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['entry'] = {}
        if block == 'type':
            groupcache[groupcounter]['varnames'] = []
        if case in ['call', 'callfun']:
            if name not in groupcache[groupcounter - 2]['externals']:
                groupcache[groupcounter - 2]['externals'].append(name)
            groupcache[groupcounter]['vars'] = copy.deepcopy(groupcache[groupcounter - 2]['vars'])
            try:
                del groupcache[groupcounter]['vars'][name][groupcache[groupcounter]['vars'][name]['attrspec'].index('external')]
            except Exception:
                pass
        if block in ['function', 'subroutine']:
            if bindcline:
                bindcdat = re.search(crackline_bindlang, bindcline)
                if bindcdat:
                    groupcache[groupcounter]['bindlang'] = {name: {}}
                    groupcache[groupcounter]['bindlang'][name]['lang'] = bindcdat.group('lang')
                    if bindcdat.group('lang_name'):
                        groupcache[groupcounter]['bindlang'][name]['name'] = bindcdat.group('lang_name')
            try:
                groupcache[groupcounter]['vars'][name] = appenddecl(groupcache[groupcounter]['vars'][name], groupcache[groupcounter - 2]['vars'][''])
            except Exception:
                pass
            if case == 'callfun':
                if result and result in groupcache[groupcounter]['vars']:
                    if not name == result:
                        groupcache[groupcounter]['vars'][name] = appenddecl(groupcache[groupcounter]['vars'][name], groupcache[groupcounter]['vars'][result])
            try:
                groupcache[groupcounter - 2]['interfaced'].append(name)
            except Exception:
                pass
        if block == 'function':
            t = typespattern[0].match(m.group('before') + ' ' + name)
            if t:
                typespec, selector, attr, edecl = cracktypespec0(t.group('this'), t.group('after'))
                updatevars(typespec, selector, attr, edecl)
        if case in ['call', 'callfun']:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
    elif case == 'entry':
        name, args, result, _ = _resolvenameargspattern(m.group('after'))
        if name is not None:
            if args:
                args = rmbadname([x.strip() for x in markoutercomma(args).split('@,@')])
            else:
                args = []
            assert result is None, repr(result)
            groupcache[groupcounter]['entry'][name] = args
            previous_context = ('entry', name, groupcounter)
    elif case == 'type':
        typespec, selector, attr, edecl = cracktypespec0(block, m.group('after'))
        last_name = updatevars(typespec, selector, attr, edecl)
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case in ['dimension', 'intent', 'optional', 'required', 'external', 'public', 'private', 'intrinsic']:
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()
        i = ll.find('::')
        if i < 0 and case == 'intent':
            i = markouterparen(ll).find('@)@') - 2
            ll = ll[:i + 1] + '::' + ll[i + 1:]
            i = ll.find('::')
            if ll[i:] == '::' and 'args' in groupcache[groupcounter]:
                outmess('All arguments will have attribute %s%s\n' % (m.group('this'), ll[:i]))
                ll = ll + ','.join(groupcache[groupcounter]['args'])
        if i < 0:
            i = 0
            pl = ''
        else:
            pl = ll[:i].strip()
            ll = ll[i + 2:]
        ch = markoutercomma(pl).split('@,@')
        if len(ch) > 1:
            pl = ch[0]
            outmess('analyzeline: cannot handle multiple attributes without type specification. Ignoring %r.\n' % ','.join(ch[1:]))
        last_name = None
        for e in [x.strip() for x in markoutercomma(ll).split('@,@')]:
            m1 = namepattern.match(e)
            if not m1:
                if case in ['public', 'private']:
                    k = ''
                else:
                    print(m.groupdict())
                    outmess('analyzeline: no name pattern found in %s statement for %s. Skipping.\n' % (case, repr(e)))
                    continue
            else:
                k = rmbadname1(m1.group('name'))
            if case in ['public', 'private'] and (k == 'operator' or k == 'assignment'):
                k += m1.group('after')
            if k not in edecl:
                edecl[k] = {}
            if case == 'dimension':
                ap = case + m1.group('after')
            if case == 'intent':
                ap = m.group('this') + pl
                if _intentcallbackpattern.match(ap):
                    if k not in groupcache[groupcounter]['args']:
                        if groupcounter > 1:
                            if '__user__' not in groupcache[groupcounter - 2]['name']:
                                outmess('analyzeline: missing __user__ module (could be nothing)\n')
                            if k != groupcache[groupcounter]['name']:
                                outmess('analyzeline: appending intent(callback) %s to %s arguments\n' % (k, groupcache[groupcounter]['name']))
                                groupcache[groupcounter]['args'].append(k)
                        else:
                            errmess('analyzeline: intent(callback) %s is ignored\n' % k)
                    else:
                        errmess('analyzeline: intent(callback) %s is already in argument list\n' % k)
            if case in ['optional', 'required', 'public', 'external', 'private', 'intrinsic']:
                ap = case
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append(ap)
            else:
                edecl[k]['attrspec'] = [ap]
            if case == 'external':
                if groupcache[groupcounter]['block'] == 'program':
                    outmess('analyzeline: ignoring program arguments\n')
                    continue
                if k not in groupcache[groupcounter]['args']:
                    continue
                if 'externals' not in groupcache[groupcounter]:
                    groupcache[groupcounter]['externals'] = []
                groupcache[groupcounter]['externals'].append(k)
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'moduleprocedure':
        groupcache[groupcounter]['implementedby'] = [x.strip() for x in m.group('after').split(',')]
    elif case == 'parameter':
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()[1:-1]
        last_name = None
        for e in markoutercomma(ll).split('@,@'):
            try:
                k, initexpr = [x.strip() for x in e.split('=')]
            except Exception:
                outmess('analyzeline: could not extract name,expr in parameter statement "%s" of "%s"\n' % (e, ll))
                continue
            params = get_parameters(edecl)
            k = rmbadname1(k)
            if k not in edecl:
                edecl[k] = {}
            if '=' in edecl[k] and (not edecl[k]['='] == initexpr):
                outmess('analyzeline: Overwriting the value of parameter "%s" ("%s") with "%s".\n' % (k, edecl[k]['='], initexpr))
            t = determineexprtype(initexpr, params)
            if t:
                if t.get('typespec') == 'real':
                    tt = list(initexpr)
                    for m in real16pattern.finditer(initexpr):
                        tt[m.start():m.end()] = list(initexpr[m.start():m.end()].lower().replace('d', 'e'))
                    initexpr = ''.join(tt)
                elif t.get('typespec') == 'complex':
                    initexpr = initexpr[1:].lower().replace('d', 'e').replace(',', '+1j*(')
            try:
                v = eval(initexpr, {}, params)
            except (SyntaxError, NameError, TypeError) as msg:
                errmess('analyzeline: Failed to evaluate %r. Ignoring: %s\n' % (initexpr, msg))
                continue
            edecl[k]['='] = repr(v)
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append('parameter')
            else:
                edecl[k]['attrspec'] = ['parameter']
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'implicit':
        if m.group('after').strip().lower() == 'none':
            groupcache[groupcounter]['implicit'] = None
        elif m.group('after'):
            if 'implicit' in groupcache[groupcounter]:
                impl = groupcache[groupcounter]['implicit']
            else:
                impl = {}
            if impl is None:
                outmess('analyzeline: Overwriting earlier "implicit none" statement.\n')
                impl = {}
            for e in markoutercomma(m.group('after')).split('@,@'):
                decl = {}
                m1 = re.match('\\s*(?P<this>.*?)\\s*(\\(\\s*(?P<after>[a-z-, ]+)\\s*\\)\\s*|)\\Z', e, re.I)
                if not m1:
                    outmess('analyzeline: could not extract info of implicit statement part "%s"\n' % e)
                    continue
                m2 = typespattern4implicit.match(m1.group('this'))
                if not m2:
                    outmess('analyzeline: could not extract types pattern of implicit statement part "%s"\n' % e)
                    continue
                typespec, selector, attr, edecl = cracktypespec0(m2.group('this'), m2.group('after'))
                kindselect, charselect, typename = cracktypespec(typespec, selector)
                decl['typespec'] = typespec
                decl['kindselector'] = kindselect
                decl['charselector'] = charselect
                decl['typename'] = typename
                for k in list(decl.keys()):
                    if not decl[k]:
                        del decl[k]
                for r in markoutercomma(m1.group('after')).split('@,@'):
                    if '-' in r:
                        try:
                            begc, endc = [x.strip() for x in r.split('-')]
                        except Exception:
                            outmess('analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement\n' % r)
                            continue
                    else:
                        begc = endc = r.strip()
                    if not len(begc) == len(endc) == 1:
                        outmess('analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement (2)\n' % r)
                        continue
                    for o in range(ord(begc), ord(endc) + 1):
                        impl[chr(o)] = decl
            groupcache[groupcounter]['implicit'] = impl
    elif case == 'data':
        ll = []
        dl = ''
        il = ''
        f = 0
        fc = 1
        inp = 0
        for c in m.group('after'):
            if not inp:
                if c == "'":
                    fc = not fc
                if c == '/' and fc:
                    f = f + 1
                    continue
            if c == '(':
                inp = inp + 1
            elif c == ')':
                inp = inp - 1
            if f == 0:
                dl = dl + c
            elif f == 1:
                il = il + c
            elif f == 2:
                dl = dl.strip()
                if dl.startswith(','):
                    dl = dl[1:].strip()
                ll.append([dl, il])
                dl = c
                il = ''
                f = 0
        if f == 2:
            dl = dl.strip()
            if dl.startswith(','):
                dl = dl[1:].strip()
            ll.append([dl, il])
        vars = groupcache[groupcounter].get('vars', {})
        last_name = None
        for l in ll:
            l[0], l[1] = (l[0].strip(), l[1].strip())
            if l[0].startswith(','):
                l[0] = l[0][1:]
            if l[0].startswith('('):
                outmess('analyzeline: implied-DO list "%s" is not supported. Skipping.\n' % l[0])
                continue
            for idx, v in enumerate(rmbadname([x.strip() for x in markoutercomma(l[0]).split('@,@')])):
                if v.startswith('('):
                    outmess('analyzeline: implied-DO list "%s" is not supported. Skipping.\n' % v)
                    continue
                if '!' in l[1]:
                    outmess('Comment line in declaration "%s" is not supported. Skipping.\n' % l[1])
                    continue
                vars.setdefault(v, {})
                vtype = vars[v].get('typespec')
                vdim = getdimension(vars[v])
                matches = re.findall('\\(.*?\\)', l[1]) if vtype == 'complex' else l[1].split(',')
                try:
                    new_val = '(/{}/)'.format(', '.join(matches)) if vdim else matches[idx]
                except IndexError:
                    if any(('*' in m for m in matches)):
                        expanded_list = []
                        for match in matches:
                            if '*' in match:
                                try:
                                    multiplier, value = match.split('*')
                                    expanded_list.extend([value.strip()] * int(multiplier))
                                except ValueError:
                                    expanded_list.append(match.strip())
                            else:
                                expanded_list.append(match.strip())
                        matches = expanded_list
                    new_val = '(/{}/)'.format(', '.join(matches)) if vdim else matches[idx]
                current_val = vars[v].get('=')
                if current_val and current_val != new_val:
                    outmess('analyzeline: changing init expression of "%s" ("%s") to "%s"\n' % (v, current_val, new_val))
                vars[v]['='] = new_val
                last_name = v
        groupcache[groupcounter]['vars'] = vars
        if last_name:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'common':
        line = m.group('after').strip()
        if not line[0] == '/':
            line = '//' + line
        cl = []
        f = 0
        bn = ''
        ol = ''
        for c in line:
            if c == '/':
                f = f + 1
                continue
            if f >= 3:
                bn = bn.strip()
                if not bn:
                    bn = '_BLNK_'
                cl.append([bn, ol])
                f = f - 2
                bn = ''
                ol = ''
            if f % 2:
                bn = bn + c
            else:
                ol = ol + c
        bn = bn.strip()
        if not bn:
            bn = '_BLNK_'
        cl.append([bn, ol])
        commonkey = {}
        if 'common' in groupcache[groupcounter]:
            commonkey = groupcache[groupcounter]['common']
        for c in cl:
            if c[0] not in commonkey:
                commonkey[c[0]] = []
            for i in [x.strip() for x in markoutercomma(c[1]).split('@,@')]:
                if i:
                    commonkey[c[0]].append(i)
        groupcache[groupcounter]['common'] = commonkey
        previous_context = ('common', bn, groupcounter)
    elif case == 'use':
        m1 = re.match('\\A\\s*(?P<name>\\b\\w+\\b)\\s*((,(\\s*\\bonly\\b\\s*:|(?P<notonly>))\\s*(?P<list>.*))|)\\s*\\Z', m.group('after'), re.I)
        if m1:
            mm = m1.groupdict()
            if 'use' not in groupcache[groupcounter]:
                groupcache[groupcounter]['use'] = {}
            name = m1.group('name')
            groupcache[groupcounter]['use'][name] = {}
            isonly = 0
            if 'list' in mm and mm['list'] is not None:
                if 'notonly' in mm and mm['notonly'] is None:
                    isonly = 1
                groupcache[groupcounter]['use'][name]['only'] = isonly
                ll = [x.strip() for x in mm['list'].split(',')]
                rl = {}
                for l in ll:
                    if '=' in l:
                        m2 = re.match('\\A\\s*(?P<local>\\b\\w+\\b)\\s*=\\s*>\\s*(?P<use>\\b\\w+\\b)\\s*\\Z', l, re.I)
                        if m2:
                            rl[m2.group('local').strip()] = m2.group('use').strip()
                        else:
                            outmess('analyzeline: Not local=>use pattern found in %s\n' % repr(l))
                    else:
                        rl[l] = l
                    groupcache[groupcounter]['use'][name]['map'] = rl
            else:
                pass
        else:
            print(m.groupdict())
            outmess('analyzeline: Could not crack the use statement.\n')
    elif case in ['f2pyenhancements']:
        if 'f2pyenhancements' not in groupcache[groupcounter]:
            groupcache[groupcounter]['f2pyenhancements'] = {}
        d = groupcache[groupcounter]['f2pyenhancements']
        if m.group('this') == 'usercode' and 'usercode' in d:
            if isinstance(d['usercode'], str):
                d['usercode'] = [d['usercode']]
            d['usercode'].append(m.group('after'))
        else:
            d[m.group('this')] = m.group('after')
    elif case == 'multiline':
        if previous_context is None:
            if verbose:
                outmess('analyzeline: No context for multiline block.\n')
            return
        gc = groupcounter
        appendmultiline(groupcache[gc], previous_context[:2], m.group('this'))
    elif verbose > 1:
        print(m.groupdict())
        outmess('analyzeline: No code implemented for line.\n')