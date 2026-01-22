from __future__ import annotations
import abc
import json
import logging
import os
import warnings
from glob import glob
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.gaussian import GaussianOutput
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Dynmat, Oszicar, Vasprun
class GaussianToComputedEntryDrone(AbstractDrone):
    """GaussianToEntryDrone assimilates directories containing Gaussian output to
    ComputedEntry/ComputedStructureEntry objects. By default, it is assumed
    that Gaussian output files have a ".log" extension.

    Note:
        Like the GaussianOutput class, this is still in early beta.
    """

    def __init__(self, inc_structure=False, parameters=None, data=None, file_extensions=('.log',)):
        """
        Args:
            inc_structure (bool): Set to True if you want
                ComputedStructureEntries to be returned instead of
                ComputedEntries.
            parameters (list): Input parameters to include. It has to be one of
                the properties supported by the GaussianOutput object. See
                pymatgen.io.gaussian.GaussianOutput. The parameters
                have to be one of python's primitive types, i.e., list, dict of
                strings and integers. If parameters is None, a default set of
                parameters will be set.
            data (list): Output data to include. Has to be one of the properties
                supported by the GaussianOutput object. The parameters have to
                be one of python's primitive types, i.e. list, dict of strings
                and integers. If data is None, a default set will be set.
            file_extensions (list):
                File extensions to be considered as Gaussian output files.
                Defaults to just the typical "log" extension.
        """
        self._inc_structure = inc_structure
        self._parameters = {'functional', 'basis_set', 'charge', 'spin_multiplicity', 'route_parameters'}
        if parameters:
            self._parameters.update(parameters)
        self._data = {'stationary_type', 'properly_terminated'}
        if data:
            self._data.update(data)
        self._file_extensions = file_extensions

    def assimilate(self, path):
        """Assimilate data in a directory path into a ComputedEntry object.

        Args:
            path: directory path

        Returns:
            ComputedEntry
        """
        try:
            gau_run = GaussianOutput(path)
        except Exception as exc:
            logger.debug(f'error in {path}: {exc}')
            return None
        param = {}
        for p in self._parameters:
            param[p] = getattr(gau_run, p)
        data = {}
        for d in self._data:
            data[d] = getattr(gau_run, d)
        if self._inc_structure:
            entry = ComputedStructureEntry(gau_run.final_structure, gau_run.final_energy, parameters=param, data=data)
        else:
            entry = ComputedEntry(gau_run.final_structure.composition, gau_run.final_energy, parameters=param, data=data)
        return entry

    def get_valid_paths(self, path):
        """Checks if path contains files with define extensions.

        Args:
            path: input path as a tuple generated from os.walk, i.e.,
                (parent, subdirs, files).

        Returns:
            List of valid dir/file paths for assimilation
        """
        parent, _subdirs, files = path
        return [os.path.join(parent, file) for file in files if os.path.splitext(file)[1] in self._file_extensions]

    def __str__(self):
        return ' GaussianToComputedEntryDrone'

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'init_args': {'inc_structure': self._inc_structure, 'parameters': self._parameters, 'data': self._data, 'file_extensions': self._file_extensions}, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict Representation.

        Returns:
            GaussianToComputedEntryDrone
        """
        return cls(**dct['init_args'])