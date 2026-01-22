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
def _get_profile_data_generator(self):
    """Get function that generates `ProfileDatum` objects.

    Returns:
      A function that generates `ProfileDatum` objects.
    """
    node_to_file_path = {}
    node_to_line_number = {}
    node_to_func_name = {}
    node_to_op_type = {}
    for op in self._graph.get_operations():
        for trace_entry in reversed(op.traceback):
            file_path = trace_entry[0]
            line_num = trace_entry[1]
            func_name = trace_entry[2]
            if not source_utils.guess_is_tensorflow_py_library(file_path):
                break
        node_to_file_path[op.name] = file_path
        node_to_line_number[op.name] = line_num
        node_to_func_name[op.name] = func_name
        node_to_op_type[op.name] = op.type

    def profile_data_generator(device_step_stats):
        for node_stats in device_step_stats.node_stats:
            if node_stats.node_name == '_SOURCE' or node_stats.node_name == '_SINK':
                continue
            yield profiling.ProfileDatum(device_step_stats.device, node_stats, node_to_file_path.get(node_stats.node_name, ''), node_to_line_number.get(node_stats.node_name, 0), node_to_func_name.get(node_stats.node_name, ''), node_to_op_type.get(node_stats.node_name, ''))
    return profile_data_generator