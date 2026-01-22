from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.TimeColumn')
def TimeColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: datetime.time | None=None, format: str | None=None, min_value: datetime.time | None=None, max_value: datetime.time | None=None, step: int | float | datetime.timedelta | None=None) -> ColumnConfig:
    """Configure a time column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for time values. This command needs to be used in
    the ``column_config`` parameter of ``st.dataframe`` or ``st.data_editor``. When
    used with ``st.data_editor``, editing will be enabled with a time picker widget.

    Parameters
    ----------

    label: str or None
        The label shown at the top of the column. If None (default),
        the column name is used.

    width: "small", "medium", "large", or None
        The display width of the column. Can be one of "small", "medium", or "large".
        If None (default), the column will be sized to fit the cell contents.

    help: str or None
        An optional tooltip that gets displayed when hovering over the column label.

    disabled: bool or None
        Whether editing should be disabled for this column. Defaults to False.

    required: bool or None
        Whether edited cells in the column need to have a value. If True, an edited cell
        can only be submitted if it has a value other than None. Defaults to False.

    default: datetime.time or None
        Specifies the default value in this column when a new row is added by the user.

    format: str or None
        A momentJS format string controlling how times are displayed. See
        `momentJS docs <https://momentjs.com/docs/#/displaying/format/>`_ for available
        formats. If None (default), uses ``HH:mm:ss``.

    min_value: datetime.time or None
        The minimum time that can be entered.
        If None (default), there will be no minimum.

    max_value: datetime.time or None
        The maximum time that can be entered.
        If None (default), there will be no maximum.

    step: int, float, datetime.timedelta, or None
        The stepping interval in seconds. If None (default), the step will be 1 second.

    Examples
    --------

    >>> from datetime import time
    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "appointment": [
    >>>             time(12, 30),
    >>>             time(18, 0),
    >>>             time(9, 10),
    >>>             time(16, 25),
    >>>         ]
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "appointment": st.column_config.TimeColumn(
    >>>             "Appointment",
    >>>             min_value=time(8, 0, 0),
    >>>             max_value=time(19, 0, 0),
    >>>             format="hh:mm a",
    >>>             step=60,
    >>>         ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-time-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=None if default is None else default.isoformat(), type_config=TimeColumnConfig(type='time', format=format, min_value=None if min_value is None else min_value.isoformat(), max_value=None if max_value is None else max_value.isoformat(), step=step.total_seconds() if isinstance(step, datetime.timedelta) else step))