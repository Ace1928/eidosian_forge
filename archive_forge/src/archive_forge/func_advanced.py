import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def advanced(canvasClass):
    """A test of figures and images."""
    canvas = canvasClass((300, 300), 'test-advanced')
    return drawAdvanced(canvas)