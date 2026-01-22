import json
import os
import re
import sys
import numpy as np
def NameListToString(name_list):
    """Converts a list of integers to the equivalent ASCII string."""
    if isinstance(name_list, str):
        return name_list
    else:
        result = ''
        if name_list is not None:
            for val in name_list:
                result = result + chr(int(val))
        return result