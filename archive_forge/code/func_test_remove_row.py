from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_remove_row():
    event_history = init_event_history(All)
    df = create_df()
    widget = QgridWidget(df=df)
    widget.remove_row(rows=[2])
    assert 2 not in widget._df.index
    assert len(widget._df) == 3
    assert event_history == [{'name': 'instance_created'}, {'name': 'json_updated', 'range': (0, 100), 'triggered_by': 'remove_row'}, {'name': 'row_removed', 'indices': [2], 'source': 'api'}]