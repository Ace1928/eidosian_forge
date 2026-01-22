from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.DateColumn')
def DateColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: datetime.date | None=None, format: str | None=None, min_value: datetime.date | None=None, max_value: datetime.date | None=None, step: int | None=None) -> ColumnConfig:
    """Configure a date column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for date values. This command needs to be used in
    the ``column_config`` parameter of ``st.dataframe`` or ``st.data_editor``. When used
    with ``st.data_editor``, editing will be enabled with a date picker widget.

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

    default: datetime.date or None
        Specifies the default value in this column when a new row is added by the user.

    format: str or None
        A momentJS format string controlling how times are displayed. See
        `momentJS docs <https://momentjs.com/docs/#/displaying/format/>`_ for available
        formats. If None (default), uses ``YYYY-MM-DD``.

    min_value: datetime.date or None
        The minimum date that can be entered.
        If None (default), there will be no minimum.

    max_value: datetime.date or None
        The maximum date that can be entered.
        If None (default), there will be no maximum.

    step: int or None
        The stepping interval in days. If None (default), the step will be 1 day.

    Examples
    --------

    >>> from datetime import date
    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "birthday": [
    >>>             date(1980, 1, 1),
    >>>             date(1990, 5, 3),
    >>>             date(1974, 5, 19),
    >>>             date(2001, 8, 17),
    >>>         ]
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "birthday": st.column_config.DateColumn(
    >>>             "Birthday",
    >>>             min_value=date(1900, 1, 1),
    >>>             max_value=date(2005, 1, 1),
    >>>             format="DD.MM.YYYY",
    >>>             step=1,
    >>>         ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-date-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=None if default is None else default.isoformat(), type_config=DateColumnConfig(type='date', format=format, min_value=None if min_value is None else min_value.isoformat(), max_value=None if max_value is None else max_value.isoformat(), step=step))