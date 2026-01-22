import os
import sys
from os.path import pardir, realpath
def _generate_posix_vars():
    """Generate the Python module containing build-time variables."""
    import pprint
    vars = {}
    makefile = get_makefile_filename()
    try:
        _parse_makefile(makefile, vars)
    except OSError as e:
        msg = f'invalid Python installation: unable to open {makefile}'
        if hasattr(e, 'strerror'):
            msg = f'{msg} ({e.strerror})'
        raise OSError(msg)
    config_h = get_config_h_filename()
    try:
        with open(config_h, encoding='utf-8') as f:
            parse_config_h(f, vars)
    except OSError as e:
        msg = f'invalid Python installation: unable to open {config_h}'
        if hasattr(e, 'strerror'):
            msg = f'{msg} ({e.strerror})'
        raise OSError(msg)
    if _PYTHON_BUILD:
        vars['BLDSHARED'] = vars['LDSHARED']
    name = _get_sysconfigdata_name()
    if 'darwin' in sys.platform:
        import types
        module = types.ModuleType(name)
        module.build_time_vars = vars
        sys.modules[name] = module
    pybuilddir = f'build/lib.{get_platform()}-{_PY_VERSION_SHORT}'
    if hasattr(sys, 'gettotalrefcount'):
        pybuilddir += '-pydebug'
    os.makedirs(pybuilddir, exist_ok=True)
    destfile = os.path.join(pybuilddir, name + '.py')
    with open(destfile, 'w', encoding='utf8') as f:
        f.write('# system configuration generated and used by the sysconfig module\n')
        f.write('build_time_vars = ')
        pprint.pprint(vars, stream=f)
    with open('pybuilddir.txt', 'w', encoding='utf8') as f:
        f.write(pybuilddir)