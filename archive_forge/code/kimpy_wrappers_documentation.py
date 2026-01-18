import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError

        See if a 'model-init' field exists in the SM metadata and, if
        so, whether it contains any entries including an "atom_style"
        command.  This is specific to LAMMPS SMs and is only required
        for using the LAMMPSrun calculator because it uses
        lammps.inputwriter to create a data file.  All other content in
        'model-init', if it exists, is ignored.
        