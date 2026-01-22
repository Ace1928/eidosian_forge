import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _get_centered_top(self, top, full_height, item_height):
    """Get the y-coordinate of the point that a figure should start at if
        its height is 'item_height' and it needs to be centered in an area that
        starts at 'top' and is 'full_height' tall."""
    return top + (full_height - item_height) / 2