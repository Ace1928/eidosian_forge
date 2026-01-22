from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class AutoFor(Loop):

    def __init__(self, target, iter_, body):
        super(AutoFor, self).__init__(body)
        self.target = target
        self.iter = iter_

    def intro_line(self):
        if self.target == '_':
            return 'for (PYTHRAN_UNUSED auto&& {0}: {1})'.format(self.target, self.iter)
        else:
            return 'for (auto&& {0}: {1})'.format(self.target, self.iter)