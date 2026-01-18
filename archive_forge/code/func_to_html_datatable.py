import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def to_html_datatable(df=None, caption=None, tableId=None, connected=True, use_to_html=False, **kwargs):
    check_table_id(tableId)
    if 'import_jquery' in kwargs:
        raise TypeError("The argument 'import_jquery' was removed in ITables v2.0. Please pass a custom 'dt_url' instead.")
    if use_to_html or (pd_style is not None and isinstance(df, pd_style.Styler)):
        return to_html_datatable_using_to_html(df=df, caption=caption, tableId=tableId, connected=connected, **kwargs)
    'Return the HTML representation of the given dataframe as an interactive datatable'
    set_default_options(kwargs, use_to_html=False)
    classes = kwargs.pop('classes')
    style = kwargs.pop('style')
    tags = kwargs.pop('tags')
    if caption is not None:
        tags = '{}<caption style="white-space: nowrap; overflow: hidden">{}</caption>'.format(tags, caption)
    showIndex = kwargs.pop('showIndex')
    if isinstance(df, (np.ndarray, np.generic)):
        df = pd.DataFrame(df)
    if isinstance(df, (pd.Series, pl.Series)):
        df = df.to_frame()
    if showIndex == 'auto':
        try:
            showIndex = df.index.name is not None or not isinstance(df.index, pd.RangeIndex)
        except AttributeError:
            showIndex = False
    maxBytes = kwargs.pop('maxBytes', 0)
    maxRows = kwargs.pop('maxRows', 0)
    maxColumns = kwargs.pop('maxColumns', pd.get_option('display.max_columns') or 0)
    warn_on_unexpected_types = kwargs.pop('warn_on_unexpected_types', False)
    df, downsampling_warning = downsample(df, max_rows=maxRows, max_columns=maxColumns, max_bytes=maxBytes)
    if downsampling_warning and 'fnInfoCallback' not in kwargs:
        kwargs['fnInfoCallback'] = JavascriptFunction("function (oSettings, iStart, iEnd, iMax, iTotal, sPre) {{ return sPre + ' ({warning})'; }}".format(warning=downsampling_warning))
    has_default_layout = kwargs['layout'] == DEFAULT_LAYOUT
    if 'dom' in kwargs:
        if opt.warn_on_dom:
            warnings.warn("The 'dom' argument has been deprecated in DataTables==2.0.", DeprecationWarning)
        if not has_default_layout:
            raise ValueError("You can pass both 'dom' and 'layout'")
        del kwargs['layout']
        has_default_layout = False
    if has_default_layout and _df_fits_in_one_page(df, kwargs):

        def filter_control(control):
            if control == 'info' and downsampling_warning:
                return control
            if control not in DEFAULT_LAYOUT_CONTROLS:
                return control
            return None
        kwargs['layout'] = {key: filter_control(control) for key, control in kwargs['layout'].items()}
    if 'buttons' in kwargs and 'layout' in kwargs and ('buttons' not in kwargs['layout'].values()):
        kwargs['layout'] = {**kwargs['layout'], 'topStart': 'buttons'}
    footer = kwargs.pop('footer')
    column_filters = kwargs.pop('column_filters')
    if column_filters == 'header':
        pass
    elif column_filters == 'footer':
        footer = True
    elif column_filters is not False:
        raise ValueError("column_filters should be either 'header', 'footer' or False, not {}".format(column_filters))
    tableId = tableId or 'itables_' + str(uuid.uuid4()).replace('-', '_')
    if isinstance(classes, list):
        classes = ' '.join(classes)
    if not showIndex:
        try:
            df = df.set_index(pd.RangeIndex(len(df.index)))
        except AttributeError:
            pass
    table_header = _table_header(df, tableId, showIndex, classes, style, tags, footer, column_filters)
    if showIndex:
        df = safe_reset_index(df)
    column_count = _column_count_in_header(table_header)
    dt_data = datatables_rows(df, column_count, warn_on_unexpected_types=warn_on_unexpected_types)
    return html_table_from_template(table_header, table_id=tableId, data=dt_data, kwargs=kwargs, connected=connected, column_filters=column_filters)