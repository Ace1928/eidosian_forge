from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def __get_closest(intersection):
    intersection.sort()
    matches = []
    for entry in intersection:
        if entry[0] == intersection[0][0]:
            matches.append(entry[2])
    return matches