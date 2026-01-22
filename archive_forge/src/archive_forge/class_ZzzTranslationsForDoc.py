import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class ZzzTranslationsForDoc(ZzzTranslations):
    _section_pat = re.compile(':\\w+:\\n\\s+')
    _indent_pat = re.compile('\\s+')

    def zzz(self, s):
        m = self._section_pat.match(s)
        if m is None:
            m = self._indent_pat.match(s)
        if m:
            return '{}zz{{{{{}}}}}'.format(m.group(0), s[m.end():])
        return 'zz{{%s}}' % s