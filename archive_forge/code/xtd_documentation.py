import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase.io.xsd import SetChild, _write_xsd_html
from ase import Atoms
Import xtd file (Materials Studio)

    Xtd files always come with arc file, and arc file
    contains all the relevant information to make atoms
    so only Arc file needs to be read
    