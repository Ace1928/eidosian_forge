import io
import os
import pickle
import sys
import runpy
import types
import warnings
from . import get_start_method, set_start_method
from . import process
from . import util
def _Django_old_layout_hack__save():
    if 'DJANGO_PROJECT_DIR' not in os.environ:
        try:
            settings_name = os.environ['DJANGO_SETTINGS_MODULE']
        except KeyError:
            return
        conf_settings = sys.modules.get('django.conf.settings')
        configured = conf_settings and conf_settings.configured
        try:
            project_name, _ = settings_name.split('.', 1)
        except ValueError:
            return
        project = __import__(project_name)
        try:
            project_dir = os.path.normpath(_module_parent_dir(project))
        except AttributeError:
            return
        if configured:
            warnings.warn(UserWarning(W_OLD_DJANGO_LAYOUT % os.path.realpath(project_dir)))
        os.environ['DJANGO_PROJECT_DIR'] = project_dir