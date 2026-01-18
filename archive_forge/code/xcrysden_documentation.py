from __future__ import annotations
from typing import TYPE_CHECKING
from pymatgen.core import Element, Structure

        Initialize a `Structure` object from a string with data in XSF format.

        Args:
            input_string: String with the structure in XSF format.
                See http://www.xcrysden.org/doc/XSF.html
            cls_: Structure class to be created. default: pymatgen structure

        Example file:
            CRYSTAL                                        see (1)
            these are primitive lattice vectors (in Angstroms)
            PRIMVEC
            0.0000000    2.7100000    2.7100000         see (2)
            2.7100000    0.0000000    2.7100000
            2.7100000    2.7100000    0.0000000

            these are conventional lattice vectors (in Angstroms)
            CONVVEC
            5.4200000    0.0000000    0.0000000         see (3)
            0.0000000    5.4200000    0.0000000
            0.0000000    0.0000000    5.4200000

            these are atomic coordinates in a primitive unit cell  (in Angstroms)
            PRIMCOORD
            2 1                                            see (4)
            16      0.0000000     0.0000000     0.0000000  see (5)
            30      1.3550000    -1.3550000    -1.3550000
        