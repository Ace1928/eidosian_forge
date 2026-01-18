from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_object_dtype_categorical():
    cat_series = pd.Series(pd.Categorical(my_object_vals, categories=my_object_vals))
    widget = show_grid(cat_series)
    constraints_enum = widget._columns[0]['constraints']['enum']
    assert not isinstance(constraints_enum[0], dict)
    assert not isinstance(constraints_enum[1], dict)
    widget._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 0, 'search_val': None})
    widget._handle_qgrid_msg_helper({'field': 0, 'filter_info': {'field': 0, 'selected': [0], 'type': 'text', 'excluded': []}, 'type': 'change_filter'})
    assert len(widget._df) == 1
    assert widget._df[0][0] == cat_series[0]