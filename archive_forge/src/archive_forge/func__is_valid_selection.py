import copy
import re
from collections import namedtuple
def _is_valid_selection(self, result, selection_list):
    if len(result) == 0 or len(selection_list) == 0:
        return False
    for item in result:
        if item not in selection_list:
            return False
    return True