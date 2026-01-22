from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
class QuotedVar(Variable):
    value_type = six.string_types

    def execute(self, interpreter):
        try:
            var = interpreter.vars[self.value()]
        except KeyError:
            raise BibTeXError('can not push undefined variable %s' % self.value())
        interpreter.push(var)