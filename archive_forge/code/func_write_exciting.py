import numpy as np
import xml.etree.ElementTree as ET
from ase.atoms import Atoms
from ase.units import Bohr
from ase.utils import writer
from xml.dom import minidom
@writer
def write_exciting(fileobj, images):
    """writes exciting input structure in XML

    Parameters
    ----------
    filename : str
        Name of file to which data should be written.
    images : Atom Object or List of Atoms objects
        This function will write the first Atoms object to file.

    Returns
    -------
    """
    root = atoms2etree(images)
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent='\t')
    fileobj.write(pretty)