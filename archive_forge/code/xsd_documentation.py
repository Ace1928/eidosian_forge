import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    connectivity: number of atoms by number of atoms matrix for connectivity
    between atoms (0 not connected, 1 connected)

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    