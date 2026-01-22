from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.util.plotting import pretty_plot
def _choose_best_x_label(self, formula, work_ion_symbol):
    if self.xaxis in {'capacity', 'capacity_grav'}:
        return 'Capacity (mAh/g)'
    if self.xaxis == 'capacity_vol':
        return 'Capacity (Ah/l)'
    formula = formula.pop() if len(formula) == 1 else None
    work_ion_symbol = work_ion_symbol.pop() if len(work_ion_symbol) == 1 else None
    if self.xaxis == 'x_form':
        if formula and work_ion_symbol:
            return f'x in {work_ion_symbol}<sub>x</sub>{formula}'
        return 'x Work Ion per Host F.U.'
    if self.xaxis == 'frac_x':
        if work_ion_symbol:
            return f'Atomic Fraction of {work_ion_symbol}'
        return 'Atomic Fraction of Working Ion'
    raise RuntimeError('No xaxis label can be determined')