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
def _flat_header(df, show_index):
    """When column filters are shown, we need to remove any column multiindex"""
    header = ''
    if show_index:
        for index in df.index.names:
            header += '<th>{}</th>'.format(index)
    for column in df.columns:
        header += '<th>{}</th>'.format(column)
    return header