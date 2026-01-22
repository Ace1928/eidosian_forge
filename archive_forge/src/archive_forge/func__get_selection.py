import copy
import re
from collections import namedtuple
def _get_selection(self, name, selection_list, default_value=None):
    result = self._get_selection_list(name, selection_list, default_value)
    if len(result) != 1:
        raise ValueError("Invalid Option Value: The option '" + name + "' can only be one of the following values:\n" + str(selection_list) + "\nYou passed in: '" + str(getattr(self.raw_options, name, None)) + "'")
    return result[0]