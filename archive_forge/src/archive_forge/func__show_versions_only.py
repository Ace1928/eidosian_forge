from functools import reduce
import sys
from os.path import dirname
def _show_versions_only():
    print('\nINSTALLED VERSIONS')
    print('------------------')
    print('Python: %d.%d.%d.%s.%s' % sys.version_info[:])
    try:
        import os
        sysname, nodename, release, version, machine = os.uname()
        print('OS: {} {} {} {}'.format(sysname, release, version, machine))
        print('byteorder: %s' % sys.byteorder)
        print('LC_ALL: %s' % os.environ.get('LC_ALL', 'None'))
        print('LANG: %s' % os.environ.get('LANG', 'None'))
    except:
        pass
    try:
        import statsmodels
        has_sm = True
    except ImportError:
        has_sm = False
    print('\nstatsmodels\n===========\n')
    if has_sm:
        print('Installed: %s' % safe_version(statsmodels))
    else:
        print('Not installed')
    print('\nRequired Dependencies\n=====================\n')
    try:
        import Cython
        print('cython: %s' % safe_version(Cython))
    except ImportError:
        print('cython: Not installed')
    try:
        import numpy
        print('numpy: %s' % safe_version(numpy, ['version', 'version']))
    except ImportError:
        print('numpy: Not installed')
    try:
        import scipy
        print('scipy: %s' % safe_version(scipy, ['version', 'version']))
    except ImportError:
        print('scipy: Not installed')
    try:
        import pandas
        print('pandas: %s' % safe_version(pandas))
    except ImportError:
        print('pandas: Not installed')
    try:
        import dateutil
        print('    dateutil: %s' % safe_version(dateutil))
    except ImportError:
        print('    dateutil: not installed')
    try:
        import patsy
        print('patsy: %s' % safe_version(patsy))
    except ImportError:
        print('patsy: Not installed')
    print('\nOptional Dependencies\n=====================\n')
    try:
        import matplotlib as mpl
        print('matplotlib: %s' % safe_version(mpl))
    except ImportError:
        print('matplotlib: Not installed')
    try:
        from cvxopt import info
        print('cvxopt: %s' % safe_version(info, 'version'))
    except ImportError:
        print('cvxopt: Not installed')
    try:
        import joblib
        print('joblib: %s ' % safe_version(joblib))
    except ImportError:
        print('joblib: Not installed')
    print('\nDeveloper Tools\n================\n')
    try:
        import IPython
        print('IPython: %s' % safe_version(IPython))
    except ImportError:
        print('IPython: Not installed')
    try:
        import jinja2
        print('    jinja2: %s' % safe_version(jinja2))
    except ImportError:
        print('    jinja2: Not installed')
    try:
        import sphinx
        print('sphinx: %s' % safe_version(sphinx))
    except ImportError:
        print('sphinx: Not installed')
    try:
        import pygments
        print('    pygments: %s' % safe_version(pygments))
    except ImportError:
        print('    pygments: Not installed')
    try:
        import pytest
        print('pytest: {} ({})'.format(safe_version(pytest), dirname(pytest.__file__)))
    except ImportError:
        print('pytest: Not installed')
    try:
        import virtualenv
        print('virtualenv: %s' % safe_version(virtualenv))
    except ImportError:
        print('virtualenv: Not installed')
    print('\n')