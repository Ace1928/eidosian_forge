import colorsys  # colour format conversions
from math import log, exp, floor, pi
import random  # for jitter values
Generate k different RBG colours evenly-space on the spiral.

        A generator returning the RGB colour space values for k
        evenly-spaced points along the defined spiral in HSV space.

        Arguments:
         - k - the number of points to return
         - offset - how far along the spiral path to start.

        