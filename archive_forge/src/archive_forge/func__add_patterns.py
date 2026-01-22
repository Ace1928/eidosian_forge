import re
from . import lazy_regex
from .trace import mutter, warning
def _add_patterns(self, patterns, translator, prefix=''):
    while patterns:
        grouped_rules = ['(%s)' % translator(pat) for pat in patterns[:99]]
        joined_rule = '{}(?:{})$'.format(prefix, '|'.join(grouped_rules))
        self._regex_patterns.append((lazy_regex.lazy_compile(joined_rule, re.UNICODE), patterns[:99]))
        patterns = patterns[99:]