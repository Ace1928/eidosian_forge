import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
@writer
def write_xsd(fd, images, connectivity=None):
    """Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    connectivity: number of atoms by number of atoms matrix for connectivity
    between atoms (0 not connected, 1 connected)

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    """
    if hasattr(images, 'get_positions'):
        images = [images]
    XSD, ATR = _write_xsd_html(images, connectivity)
    rough_string = ET.tostring(XSD, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    Document = reparsed.toprettyxml(indent='\t')
    fd.write(Document)