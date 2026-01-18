from ase.atoms import Atoms
from ase.units import Bohr
from re import compile

    Function to write a sys file.

    fileobj: file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    