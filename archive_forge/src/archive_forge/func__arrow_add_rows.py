from __future__ import annotations
import sys
from contextvars import ContextVar
from copy import deepcopy
from typing import (
from streamlit import (
from streamlit.cursor import Cursor
from streamlit.elements.alert import AlertMixin
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import ArrowMixin
from streamlit.elements.arrow_altair import ArrowAltairMixin, prep_data
from streamlit.elements.arrow_vega_lite import ArrowVegaLiteMixin
from streamlit.elements.balloons import BalloonsMixin
from streamlit.elements.bokeh_chart import BokehMixin
from streamlit.elements.code import CodeMixin
from streamlit.elements.deck_gl_json_chart import PydeckMixin
from streamlit.elements.doc_string import HelpMixin
from streamlit.elements.empty import EmptyMixin
from streamlit.elements.exception import ExceptionMixin
from streamlit.elements.form import FormData, FormMixin, current_form_id
from streamlit.elements.graphviz_chart import GraphvizMixin
from streamlit.elements.heading import HeadingMixin
from streamlit.elements.html import HtmlMixin
from streamlit.elements.iframe import IframeMixin
from streamlit.elements.image import ImageMixin
from streamlit.elements.json import JsonMixin
from streamlit.elements.layouts import LayoutsMixin
from streamlit.elements.map import MapMixin
from streamlit.elements.markdown import MarkdownMixin
from streamlit.elements.media import MediaMixin
from streamlit.elements.metric import MetricMixin
from streamlit.elements.plotly_chart import PlotlyMixin
from streamlit.elements.progress import ProgressMixin
from streamlit.elements.pyplot import PyplotMixin
from streamlit.elements.snow import SnowMixin
from streamlit.elements.text import TextMixin
from streamlit.elements.toast import ToastMixin
from streamlit.elements.widgets.button import ButtonMixin
from streamlit.elements.widgets.camera_input import CameraInputMixin
from streamlit.elements.widgets.chat import ChatMixin
from streamlit.elements.widgets.checkbox import CheckboxMixin
from streamlit.elements.widgets.color_picker import ColorPickerMixin
from streamlit.elements.widgets.data_editor import DataEditorMixin
from streamlit.elements.widgets.file_uploader import FileUploaderMixin
from streamlit.elements.widgets.multiselect import MultiSelectMixin
from streamlit.elements.widgets.number_input import NumberInputMixin
from streamlit.elements.widgets.radio import RadioMixin
from streamlit.elements.widgets.select_slider import SelectSliderMixin
from streamlit.elements.widgets.selectbox import SelectboxMixin
from streamlit.elements.widgets.slider import SliderMixin
from streamlit.elements.widgets.text_widgets import TextWidgetsMixin
from streamlit.elements.widgets.time_widgets import TimeWidgetsMixin
from streamlit.elements.write import WriteMixin
from streamlit.errors import NoSessionContext, StreamlitAPIException
from streamlit.proto import Block_pb2, ForwardMsg_pb2
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import NoValue
def _arrow_add_rows(self: DG, data: Data=None, **kwargs: DataFrame | npt.NDArray[Any] | Iterable[Any] | dict[Hashable, Any] | None) -> DG | None:
    """Concatenate a dataframe to the bottom of the current one.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict, or None
            Table to concat. Optional.

        **kwargs : pandas.DataFrame, numpy.ndarray, Iterable, dict, or None
            The named dataset to concat. Optional. You can only pass in 1
            dataset (including the one in the data parameter).

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df1 = pd.DataFrame(
        ...    np.random.randn(50, 20),
        ...    columns=('col %d' % i for i in range(20)))
        ...
        >>> my_table = st.table(df1)
        >>>
        >>> df2 = pd.DataFrame(
        ...    np.random.randn(50, 20),
        ...    columns=('col %d' % i for i in range(20)))
        ...
        >>> my_table._arrow_add_rows(df2)
        >>> # Now the table shown in the Streamlit app contains the data for
        >>> # df1 followed by the data for df2.

        You can do the same thing with plots. For example, if you want to add
        more data to a line chart:

        >>> # Assuming df1 and df2 from the example above still exist...
        >>> my_chart = st.line_chart(df1)
        >>> my_chart._arrow_add_rows(df2)
        >>> # Now the chart shown in the Streamlit app contains the data for
        >>> # df1 followed by the data for df2.

        And for plots whose datasets are named, you can pass the data with a
        keyword argument where the key is the name:

        >>> my_chart = st._arrow_vega_lite_chart({
        ...     'mark': 'line',
        ...     'encoding': {'x': 'a', 'y': 'b'},
        ...     'datasets': {
        ...       'some_fancy_name': df1,  # <-- named dataset
        ...      },
        ...     'data': {'name': 'some_fancy_name'},
        ... }),
        >>> my_chart._arrow_add_rows(some_fancy_name=df2)  # <-- name used as keyword

        """
    if self._root_container is None or self._cursor is None:
        return self
    if not self._cursor.is_locked:
        raise StreamlitAPIException('Only existing elements can `add_rows`.')
    if data is not None and len(kwargs) == 0:
        name = ''
    elif len(kwargs) == 1:
        name, data = kwargs.popitem()
    else:
        raise StreamlitAPIException('Wrong number of arguments to add_rows().Command requires exactly one dataset')
    if self._cursor.props['delta_type'] in ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES and self._cursor.props['add_rows_metadata'].last_index is None:
        st_method_name = self._cursor.props['delta_type'].replace('arrow_', '')
        st_method = getattr(self, st_method_name)
        st_method(data, **kwargs)
        return None
    new_data, self._cursor.props['add_rows_metadata'] = _prep_data_for_add_rows(data, self._cursor.props['delta_type'], self._cursor.props['add_rows_metadata'])
    msg = ForwardMsg_pb2.ForwardMsg()
    msg.metadata.delta_path[:] = self._cursor.delta_path
    import streamlit.elements.arrow as arrow_proto
    default_uuid = str(hash(self._get_delta_path_str()))
    arrow_proto.marshall(msg.delta.arrow_add_rows.data, new_data, default_uuid)
    if name:
        msg.delta.arrow_add_rows.name = name
        msg.delta.arrow_add_rows.has_name = True
    _enqueue_message(msg)
    return self