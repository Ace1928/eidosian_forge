import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _get_domain_callback(self):
    if isinstance(self.data, Var._ComponentDataClass):
        return str(self.data.domain)
    if isinstance(self.data, (BooleanVar, BooleanVar._ComponentDataClass)):
        return 'BooleanVar'
    return None