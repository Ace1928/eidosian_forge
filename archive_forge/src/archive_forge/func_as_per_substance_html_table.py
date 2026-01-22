from collections import OrderedDict
from ..chemistry import Substance
from .numbers import number_to_scientific_html
def as_per_substance_html_table(cont, substances=None, header=None, substance_factory=Substance.from_formula):
    """ """
    if substances is None:
        substances = OrderedDict([(k, substance_factory(k)) for k in cont])

    def _elem(k):
        try:
            return cont[k]
        except (IndexError, TypeError):
            return cont[list(substances.keys()).index(k)]
    rows = [(v.html_name, number_to_scientific_html(_elem(k))) for k, v in substances.items()]
    return Table(rows, ['Substance', header or ''])