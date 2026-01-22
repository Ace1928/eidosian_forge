import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _set_active_callback(self, val):
    if not val or val in ['False', 'false', '0', 'f', 'F']:
        val = False
    else:
        val = True
    try:
        if val:
            self.data.activate()
        else:
            self.data.deactivate()
    except:
        return