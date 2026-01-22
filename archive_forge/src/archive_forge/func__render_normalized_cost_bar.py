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
def _render_normalized_cost_bar(self, cost, max_cost, length):
    """Render a text bar representing a normalized cost.

    Args:
      cost: the absolute value of the cost.
      max_cost: the maximum cost value to normalize the absolute cost with.
      length: (int) length of the cost bar, in number of characters, excluding
        the brackets on the two ends.

    Returns:
      An instance of debugger_cli_common.RichTextLine.
    """
    num_ticks = int(np.ceil(float(cost) / max_cost * length))
    num_ticks = num_ticks or 1
    output = RL('[', font_attr=self._LINE_COST_ATTR)
    output += RL('|' * num_ticks + ' ' * (length - num_ticks), font_attr=['bold', self._LINE_COST_ATTR])
    output += RL(']', font_attr=self._LINE_COST_ATTR)
    return output