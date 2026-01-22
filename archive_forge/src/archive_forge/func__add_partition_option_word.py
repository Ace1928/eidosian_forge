import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _add_partition_option_word(self, directive):
    if directive == 'PARTITION BY' or directive == 'SUBPARTITION BY':
        regex = '(?<!\\S)(?P<directive>%s)%s(?P<val>\\w+.*)' % (re.escape(directive), self._optional_equals)
    elif directive == 'SUBPARTITIONS' or directive == 'PARTITIONS':
        regex = '(?<!\\S)(?P<directive>%s)%s(?P<val>\\d+)' % (re.escape(directive), self._optional_equals)
    else:
        regex = '(?<!\\S)(?P<directive>%s)(?!\\S)' % (re.escape(directive),)
    self._pr_options.append(_pr_compile(regex))