from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
def _get_channel_names(meta, channel_numbers, channel_naming='$PnS'):
    """Get list of channel names.

    Credit: https://github.com/eyurtsev/fcsparser/blob/master/fcsparser/api.py

    Raises
    ------
    RuntimeWarning
        Warns if the names are not unique.
    """
    names_n = _channel_names_from_meta(meta, channel_numbers, 'N')
    names_s = _channel_names_from_meta(meta, channel_numbers, 'S')
    if channel_naming == '$PnS':
        channel_names, channel_names_alternate = (names_s, names_n)
    elif channel_naming == '$PnN':
        channel_names, channel_names_alternate = (names_n, names_s)
    else:
        raise ValueError("Expected channel_naming in ['$PnS', '$PnN']. Got '{}'".format(channel_naming))
    if len(channel_names) == 0:
        channel_names = channel_names_alternate
    if len(set(channel_names)) != len(channel_names):
        warnings.warn('The default channel names (defined by the {} parameter in the FCS file) were not unique. To avoid problems in downstream analysis, the channel names have been switched to the alternate channel names defined in the FCS file. To avoid seeing this warning message, explicitly instruct the FCS parser to use the alternate channel names by specifying the channel_naming parameter.'.format(channel_naming), RuntimeWarning)
        channel_names = channel_names_alternate
    return channel_names