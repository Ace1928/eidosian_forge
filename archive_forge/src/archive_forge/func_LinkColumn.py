from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.LinkColumn')
def LinkColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: str | None=None, max_chars: int | None=None, validate: str | None=None, display_text: str | None=None) -> ColumnConfig:
    """Configure a link column in ``st.dataframe`` or ``st.data_editor``.

    The cell values need to be string and will be shown as clickable links.
    This command needs to be used in the column_config parameter of ``st.dataframe``
    or ``st.data_editor``. When used with ``st.data_editor``, editing will be enabled
    with a text input widget.

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

    default: str or None
        Specifies the default value in this column when a new row is added by the user.

    max_chars: int or None
        The maximum number of characters that can be entered. If None (default),
        there will be no maximum.

    validate: str or None
        A regular expression (JS flavor, e.g. ``"^https://.+$"``) that edited values are validated against.
        If the input is invalid, it will not be submitted.

    display_text: str or None
        The text that is displayed in the cell. Can be one of:

        * ``None`` (default) to display the URL itself.

        * A string that is displayed in every cell, e.g. ``"Open link"``.

        * A regular expression (JS flavor, detected by usage of parentheses)
          to extract a part of the URL via a capture group, e.g. ``"https://(.*?)\\.example\\.com"``
          to extract the display text "foo" from the URL "\\https://foo.example.com".

        For more complex cases, you may use `Pandas Styler's format         <https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.format.html>`_
        function on the underlying dataframe. Note that this makes the app slow,
        doesn't work with editable columns, and might be removed in the future.

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "apps": [
    >>>             "https://roadmap.streamlit.app",
    >>>             "https://extras.streamlit.app",
    >>>             "https://issues.streamlit.app",
    >>>             "https://30days.streamlit.app",
    >>>         ],
    >>>         "creator": [
    >>>             "https://github.com/streamlit",
    >>>             "https://github.com/arnaudmiribel",
    >>>             "https://github.com/streamlit",
    >>>             "https://github.com/streamlit",
    >>>         ],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "apps": st.column_config.LinkColumn(
    >>>             "Trending apps",
    >>>             help="The top trending Streamlit apps",
    >>>             validate="^https://[a-z]+\\.streamlit\\.app$",
    >>>             max_chars=100,
    >>>             display_text="https://(.*?)\\.streamlit\\.app"
    >>>         ),
    >>>         "creator": st.column_config.LinkColumn(
    >>>             "App Creator", display_text="Open profile"
    >>>         ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-link-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=default, type_config=LinkColumnConfig(type='link', max_chars=max_chars, validate=validate, display_text=display_text))