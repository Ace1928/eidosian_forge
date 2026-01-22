import argparse
import os
import re
import numpy as np
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import profiling
from tensorflow.python.debug.lib import source_utils
def _measure_list_profile_column_widths(self, profile_data):
    """Determine the maximum column widths for each data list.

    Args:
      profile_data: list of ProfileDatum objects.

    Returns:
      List of column widths in the same order as columns in data.
    """
    num_columns = len(profile_data.column_names())
    widths = [len(column_name) for column_name in profile_data.column_names()]
    for row in range(profile_data.row_count()):
        for col in range(num_columns):
            widths[col] = max(widths[col], len(str(profile_data.row_values(row)[col])) + 2)
    return widths