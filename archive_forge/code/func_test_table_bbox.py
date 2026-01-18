import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
@check_figures_equal(extensions=['png'])
def test_table_bbox(fig_test, fig_ref):
    data = [[2, 3], [4, 5]]
    col_labels = ('Foo', 'Bar')
    row_labels = ('Ada', 'Bob')
    cell_text = [[f'{x}' for x in row] for row in data]
    ax_list = fig_test.subplots()
    ax_list.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    ax_bbox = fig_ref.subplots()
    ax_bbox.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center', bbox=Bbox.from_extents(0.1, 0.2, 0.9, 0.8))