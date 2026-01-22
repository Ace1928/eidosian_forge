from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.flags import _exceptions
def _get_input_to_checker_function(self, flag_values):
    """Given flag values, returns the input to be given to checker.

    Args:
      flag_values: flags.FlagValues, the FlagValues instance to get flags from.
    Returns:
      dict, with keys() being self.lag_names, and value for each key
      being the value of the corresponding flag (string, boolean, etc).
    """
    return dict(([key, flag_values[key].value] for key in self.flag_names))