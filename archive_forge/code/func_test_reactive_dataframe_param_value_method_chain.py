import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_dataframe_param_value_method_chain(dataframe):
    P = Parameters(string='str')
    dfi = rx(dataframe).groupby(P.param.string)[['float']].mean().reset_index()
    pd.testing.assert_frame_equal(dfi.rx.value, dataframe.groupby('str')[['float']].mean().reset_index())
    P.string = 'int'
    pd.testing.assert_frame_equal(dfi.rx.value, dataframe.groupby('int')[['float']].mean().reset_index())