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
def _print_feed_handler(self, args, screen_info=None):
    np_printoptions = cli_shared.numpy_printoptions_from_screen_info(screen_info)
    if not self._feed_dict:
        return cli_shared.error('The feed_dict of the current run is None or empty.')
    parsed = self._argparsers['print_feed'].parse_args(args)
    tensor_name, tensor_slicing = command_parser.parse_tensor_name_with_slicing(parsed.tensor_name)
    feed_key = None
    feed_value = None
    for key in self._feed_dict:
        key_name = common.get_graph_element_name(key)
        if key_name == tensor_name:
            feed_key = key_name
            feed_value = self._feed_dict[key]
            break
    if feed_key is None:
        return cli_shared.error('The feed_dict of the current run does not contain the key %s' % tensor_name)
    else:
        return cli_shared.format_tensor(feed_value, feed_key + ' (feed)', np_printoptions, print_all=parsed.print_all, tensor_slicing=tensor_slicing, highlight_options=cli_shared.parse_ranges_highlight(parsed.ranges), include_numeric_summary=parsed.numeric_summary)