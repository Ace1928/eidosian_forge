from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (

    Check for a per-command value or override of the given configuration key
    and return it if it exists. Otherwise return the global configuration value
    for that key.
    