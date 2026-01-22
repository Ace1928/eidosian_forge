import argparse
import os
import sys
import tempfile
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.lib.io import file_io
def _run_handler(self, args, screen_info=None):
    """Command handler for "run" command during on-run-start."""
    del screen_info
    parsed = self._argparsers['run'].parse_args(args)
    parsed.node_name_filter = parsed.node_name_filter or None
    parsed.op_type_filter = parsed.op_type_filter or None
    parsed.tensor_dtype_filter = parsed.tensor_dtype_filter or None
    if parsed.filter_exclude_node_names and (not parsed.till_filter_pass):
        raise ValueError('The --filter_exclude_node_names (or -feon) flag is valid only if the --till_filter_pass (or -f) flag is used.')
    if parsed.profile:
        raise debugger_cli_common.CommandLineExit(exit_token=framework.OnRunStartResponse(framework.OnRunStartAction.PROFILE_RUN, []))
    self._skip_debug = parsed.no_debug
    self._run_through_times = parsed.times
    if parsed.times > 1 or parsed.no_debug:
        action = framework.OnRunStartAction.NON_DEBUG_RUN
        debug_urls = []
    else:
        action = framework.OnRunStartAction.DEBUG_RUN
        debug_urls = self._get_run_debug_urls()
    run_start_response = framework.OnRunStartResponse(action, debug_urls, node_name_regex_allowlist=parsed.node_name_filter, op_type_regex_allowlist=parsed.op_type_filter, tensor_dtype_regex_allowlist=parsed.tensor_dtype_filter)
    if parsed.till_filter_pass:
        if parsed.till_filter_pass in self._tensor_filters:
            action = framework.OnRunStartAction.DEBUG_RUN
            self._active_tensor_filter = parsed.till_filter_pass
            self._active_filter_exclude_node_names = parsed.filter_exclude_node_names
            self._active_tensor_filter_run_start_response = run_start_response
        else:
            return debugger_cli_common.RichTextLines(['ERROR: tensor filter "%s" does not exist.' % parsed.till_filter_pass])
    raise debugger_cli_common.CommandLineExit(exit_token=run_start_response)