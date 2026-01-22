from collections import defaultdict
import param
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table as mpl_Table
from .element import ElementPlot
from .plot import mpl_rc_context
def _cell_value(self, element, row, col):
    summarize = element.rows > self.max_rows
    half_rows = self.max_rows // 2
    if summarize and row == half_rows:
        cell_text = '...'
    else:
        if summarize and row > half_rows:
            row = element.rows - self.max_rows + row
        cell_text = element.pprint_cell(row, col)
        if len(cell_text) > self.max_value_len:
            cell_text = cell_text[:self.max_value_len - 3] + '...'
    return cell_text