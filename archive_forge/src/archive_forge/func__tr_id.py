from .numbers import number_to_scientific_html
from .string import StrPrinter
def _tr_id(self, rsys, i):
    return 'chempy_%d_%d' % (id(rsys), i)