import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
def text_array_to_html(text_arr, enable_markdown):
    """Take a numpy.ndarray containing strings, and convert it into html.

    If the ndarray contains a single scalar string, that string is converted to
    html via our sanitized markdown parser. If it contains an array of strings,
    the strings are individually converted to html and then composed into a table
    using make_table. If the array contains dimensionality greater than 2,
    all but two of the dimensions are removed, and a warning message is prefixed
    to the table.

    Args:
      text_arr: A numpy.ndarray containing strings.
      enable_markdown: boolean, whether to enable Markdown

    Returns:
      The array converted to html.
    """
    if not text_arr.shape:
        if enable_markdown:
            return plugin_util.markdown_to_safe_html(text_arr.item())
        else:
            return plugin_util.safe_html(text_arr.item())
    warning = ''
    if len(text_arr.shape) > 2:
        warning = plugin_util.markdown_to_safe_html(WARNING_TEMPLATE % len(text_arr.shape))
        text_arr = reduce_to_2d(text_arr)
    if enable_markdown:
        table = plugin_util.markdowns_to_safe_html(text_arr.reshape(-1), lambda xs: make_table(np.array(xs).reshape(text_arr.shape)))
    else:
        decode = lambda bs: bs.decode('utf-8') if isinstance(bs, bytes) else bs
        text_arr_str = np.array([decode(bs) for bs in text_arr.reshape(-1)]).reshape(text_arr.shape)
        table = plugin_util.safe_html(make_table(text_arr_str))
    return warning + table