import copy
import re
from collections import namedtuple
def _get_selection_list(self, name, selection_list, default_value=None):
    if not selection_list:
        raise ValueError('Selection list cannot be empty.')
    default_value = default_value or [selection_list[0]]
    if not self._is_valid_selection(default_value, selection_list):
        raise ValueError('Invalid Default Value!')
    result = self._get_array(name, default_value)
    if not self._is_valid_selection(result, selection_list):
        raise ValueError("Invalid Option Value: The option '" + name + "' can contain only the following values:\n" + str(selection_list) + "\nYou passed in: '" + str(getattr(self.raw_options, name, None)) + "'")
    return result