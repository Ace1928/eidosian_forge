import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
def _get_options_string(self, options=None):
    if options is None:
        options = self.options
    ans = []
    for key in options:
        val = options[key]
        if isinstance(val, str) and ' ' in val:
            ans.append('%s="%s"' % (str(key), str(val)))
        else:
            ans.append('%s=%s' % (str(key), str(val)))
    return ' '.join(ans)