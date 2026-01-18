import ase.io
from .parse_lammps_data_file import lammpsdata_file_extracted_sections
from .comparison import compare_with_pytest_approx

Use lammpsdata module to create an Atoms object from a lammps data file
and checks that the cell, mass, positions, and velocities match the
values that are parsed directly from the data file.

NOTE: This test currently only works when using a lammps data file
containing a single atomic species
