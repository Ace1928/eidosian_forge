import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _add_option_word(self, directive):
    regex = '(?P<directive>%s)%s(?P<val>\\w+)' % (re.escape(directive), self._optional_equals)
    self._pr_options.append(_pr_compile(regex))