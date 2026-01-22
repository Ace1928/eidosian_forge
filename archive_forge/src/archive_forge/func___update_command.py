import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def __update_command(self, command=None, aims_command=None, outfilename=None):
    """
        Abstracted generic setter routine for a dynamic behavior of "command".

        The command that is actually called on the command line and enters the
        base class, is <command> = <aims_command> > <outfilename>.

        This new scheme has been introduced in order to conveniently change the
        outfile name from the outside while automatically updating the
        <command> member variable.

        Obiously, changing <command> conflicts with changing <aims_command>
        and/or <outfilename>, which thus raises a <ValueError>. This should,
        however, not happen if this routine is not used outside the property
        definitions.

        Parameters
        ----------
        command : str
            The full command as executed to run FHI-aims. This includes
            any potential mpiexec call, as well as the redirection of stdout.
            For instance "mpiexec -np 4 aims.x > aims.out".

        aims_command : str
            The full command as executed to run FHI-aims *without* the
            redirection to stdout. For instance "mpiexec -np 4 aims.x"

        outfilename : str
            The file (incl. path) to which stdout is redirected.
        """
    if command:
        if aims_command:
            raise ValueError('Cannot specify "command" and "aims_command" simultaneously.')
        if outfilename:
            raise ValueError('Cannot specify "command" and "outfilename" simultaneously.')
        command_spl = command.split('>')
        if len(command_spl) > 1:
            self.__aims_command = command_spl[0].strip()
            self.__outfilename = command_spl[-1].strip()
        else:
            self.__aims_command = command.strip()
            self.__outfilename = Aims.__outfilename_default
    else:
        if aims_command is not None:
            self.__aims_command = aims_command
        elif outfilename is None:
            return
        if outfilename is not None:
            self.__outfilename = outfilename
        elif not self.outfilename:
            self.__outfilename = Aims.__outfilename_default
    self.__command = '{0:s} > {1:s}'.format(self.aims_command, self.outfilename)