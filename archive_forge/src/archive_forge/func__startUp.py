from functools import partial
def _startUp():
    """This function allows easy resetting to the global defaults
    If the environment contains 'RL_xxx' then we use the value
    else we use the given default"""
    import os, sys
    global sys_version, _unset_
    sys_version = sys.version.split()[0]
    from reportlab.lib import pagesizes
    from reportlab.lib.utils import rl_isdir
    if _SAVED == {}:
        _unset_ = getattr(sys, '_rl_config__unset_', None)
        if _unset_ is None:

            class _unset_:
                pass
            sys._rl_config__unset_ = _unset_ = _unset_()
        global __all__
        A = list(__all__)
        for k, v in _DEFAULTS.items():
            _SAVED[k] = globals()[k] = v
            if k not in __all__:
                A.append(k)
        __all__ = tuple(A)
    import reportlab
    D = {'REPORTLAB_DIR': os.path.abspath(os.path.dirname(reportlab.__file__)), 'CWD': os.getcwd(), 'disk': os.getcwd().split(':')[0], 'sys_version': sys_version, 'XDG_DATA_HOME': os.environ.get('XDG_DATA_HOME', '~/.local/share')}
    for k in _SAVED:
        if k.endswith('SearchPath'):
            P = []
            for p in _SAVED[k]:
                d = (p % D).replace('/', os.sep)
                if '~' in d:
                    try:
                        d = os.path.expanduser(d)
                    except (KeyError, ImportError):
                        continue
                if rl_isdir(d):
                    P.append(d)
            _setOpt(k, os.pathsep.join(P), lambda x: x.split(os.pathsep))
            globals()[k] = list(filter(rl_isdir, globals()[k]))
        else:
            v = _SAVED[k]
            if isinstance(v, (int, float)):
                conv = type(v)
            elif k == 'defaultPageSize':
                conv = lambda v, M=pagesizes: getattr(M, v)
            elif k in ('trustedHosts', 'trustedSchemes'):
                conv = lambda v: None if v is None else [y for y in [x.strip() for x in v.split(',')] if y] if isinstance(v, str) else v
            else:
                conv = None
            _setOpt(k, v, conv)