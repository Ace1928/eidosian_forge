import os
import sys
from os.path import pardir, realpath
def _parse_makefile(filename, vars=None, keep_unresolved=True):
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    import re
    if vars is None:
        vars = {}
    done = {}
    notdone = {}
    with open(filename, encoding=sys.getfilesystemencoding(), errors='surrogateescape') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        m = re.match(_variable_rx, line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            tmpv = v.replace('$$', '')
            if '$' in tmpv:
                notdone[n] = v
            else:
                try:
                    if n in _ALWAYS_STR:
                        raise ValueError
                    v = int(v)
                except ValueError:
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v
    variables = list(notdone.keys())
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')
    while len(variables) > 0:
        for name in tuple(variables):
            value = notdone[name]
            m1 = re.search(_findvar1_rx, value)
            m2 = re.search(_findvar2_rx, value)
            if m1 and m2:
                m = m1 if m1.start() < m2.start() else m2
            else:
                m = m1 if m1 else m2
            if m is not None:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    found = False
                elif n in os.environ:
                    item = os.environ[n]
                elif n in renamed_variables:
                    if name.startswith('PY_') and name[3:] in renamed_variables:
                        item = ''
                    elif 'PY_' + n in notdone:
                        found = False
                    else:
                        item = str(done['PY_' + n])
                else:
                    done[n] = item = ''
                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if '$' in after:
                        notdone[name] = value
                    else:
                        try:
                            if name in _ALWAYS_STR:
                                raise ValueError
                            value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        variables.remove(name)
                        if name.startswith('PY_') and name[3:] in renamed_variables:
                            name = name[3:]
                            if name not in done:
                                done[name] = value
            else:
                if keep_unresolved:
                    done[name] = value
                variables.remove(name)
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()
    vars.update(done)
    return vars