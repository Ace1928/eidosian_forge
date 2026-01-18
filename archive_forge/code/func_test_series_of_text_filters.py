from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_series_of_text_filters():
    view = QgridWidget(df=create_df())
    view._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 'E', 'search_val': None})
    view._handle_qgrid_msg_helper({'field': 'E', 'filter_info': {'field': 'E', 'selected': [0, 1], 'type': 'text', 'excluded': []}, 'type': 'change_filter'})
    filtered_df = view.get_changed_df()
    assert len(filtered_df) == 2
    view._handle_qgrid_msg_helper({'field': 'E', 'filter_info': {'field': 'E', 'selected': None, 'type': 'text', 'excluded': []}, 'type': 'change_filter'})
    view._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 'F', 'search_val': None})
    view._handle_qgrid_msg_helper({'field': 'F', 'filter_info': {'field': 'F', 'selected': [0, 1], 'type': 'text', 'excluded': []}, 'type': 'change_filter'})
    filtered_df = view.get_changed_df()
    assert len(filtered_df) == 2