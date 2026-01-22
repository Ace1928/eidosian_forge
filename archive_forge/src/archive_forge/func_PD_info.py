import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def PD_info(self):
    """
        Displays a PD code as a list of 4-tuples.
        """
    code = self.PD_code()
    if code:
        self.write_text(('PD: %s' % code).replace(', ', ','))