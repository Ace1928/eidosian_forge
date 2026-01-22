import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
def _get_testable_qt_backends():
    envs = []
    for deps, env in [([qt_api], {'MPLBACKEND': 'qtagg', 'QT_API': qt_api}) for qt_api in ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()):
            reason = '$DISPLAY and $WAYLAND_DISPLAY are unset'
        elif missing:
            reason = '{} cannot be imported'.format(', '.join(missing))
        elif env['MPLBACKEND'] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = 'macosx backend fails on Azure'
        marks = []
        if reason:
            marks.append(pytest.mark.skip(reason=f'Skipping {env} because {reason}'))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs