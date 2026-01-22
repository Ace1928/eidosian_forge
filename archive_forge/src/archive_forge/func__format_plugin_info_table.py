from .manage_plugins import *
from .sift import *
from .collection import *
from ._io import *
from ._image_stack import *
def _format_plugin_info_table(info_table, column_lengths):
    """Add separators and column titles to plugin info table."""
    info_table.insert(0, _separator('=', column_lengths))
    info_table.insert(1, ('Plugin', 'Description'))
    info_table.insert(2, _separator('-', column_lengths))
    info_table.append(_separator('=', column_lengths))