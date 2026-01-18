from ase.atoms import Atoms
import pytest

    If 'create_atoms' hasn't been given the appropriate 'remap yes' option,
    atoms falling outside of a periodic cell are not actually created.  The
    lammpsrun calculator will then look at the thermo output and determine a
    discrepancy the number of atoms reported compared to the length of the
    ASE Atoms object and raise a RuntimeError.  This problem can only
    possibly arise when the 'no_data_file' option for the calculator is set
    to True.  Furthermore, note that if atoms fall outside of the box along
    non-periodic dimensions, create_atoms is going to refuse to create them
    no matter what, so you simply can't use the 'no_data_file' option if you
    want to allow for that scenario.
    