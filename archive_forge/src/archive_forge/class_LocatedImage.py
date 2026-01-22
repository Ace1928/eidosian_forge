import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
class LocatedImage(collections.namedtuple('LocatedImage', 'image, extent')):
    """
    Define an image and associated extent in the form:
       ``image, (min_x, max_x, min_y, max_y)``

    """