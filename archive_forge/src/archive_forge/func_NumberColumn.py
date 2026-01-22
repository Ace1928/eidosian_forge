from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.NumberColumn')
def NumberColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: int | float | None=None, format: str | None=None, min_value: int | float | None=None, max_value: int | float | None=None, step: int | float | None=None) -> ColumnConfig:
    """Configure a number column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for integer and float values. This command needs to
    be used in the ``column_config`` parameter of ``st.dataframe`` or ``st.data_editor``.
    When used with ``st.data_editor``, editing will be enabled with a numeric input widget.

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

    default: int, float, or None
        Specifies the default value in this column when a new row is added by the user.

    format : str or None
        A printf-style format string controlling how numbers are displayed.
        This does not impact the return value. Valid formatters: %d %e %f %g %i %u.
        You can also add prefixes and suffixes, e.g. ``"$ %.2f"`` to show a dollar prefix.

    min_value : int, float, or None
        The minimum value that can be entered.
        If None (default), there will be no minimum.

    max_value : int, float, or None
        The maximum value that can be entered.
        If None (default), there will be no maximum.

    step: int, float, or None
        The stepping interval. Specifies the precision of numbers that can be entered.
        If None (default), uses 1 for integers and unrestricted precision for floats.

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "price": [20, 950, 250, 500],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "price": st.column_config.NumberColumn(
    >>>             "Price (in USD)",
    >>>             help="The price of the product in USD",
    >>>             min_value=0,
    >>>             max_value=1000,
    >>>             step=1,
    >>>             format="$%d",
    >>>         )
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-number-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=default, type_config=NumberColumnConfig(type='number', min_value=min_value, max_value=max_value, format=format, step=step))