from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
class BasicAbinitInput(AbstractInput, MSONable):
    """This object stores the ABINIT variables for a single dataset."""
    Error = BasicAbinitInputError

    def __init__(self, structure, pseudos: str | list[str] | list[Pseudo] | PseudoTable, pseudo_dir=None, comment=None, abi_args=None, abi_kwargs=None):
        """
        Args:
            structure: Parameters defining the crystalline structure. Accepts |Structure| object
            file with structure (CIF, netcdf file, ...) or dictionary with ABINIT geo variables.
            pseudos: Pseudopotentials to be used for the calculation. Accepts: string or list of strings
                with the name of the pseudopotential files, list of |Pseudo| objects
                or |PseudoTable| object.
            pseudo_dir: Name of the directory where the pseudopotential files are located.
            ndtset: Number of datasets.
            comment: Optional string with a comment that will be placed at the beginning of the file.
            abi_args: list of tuples (key, value) with the initial set of variables. Default: Empty
            abi_kwargs: Dictionary with the initial set of variables. Default: Empty.
        """
        abi_args = abi_args or []
        for key, _value in abi_args:
            self._check_varname(key)
        abi_kwargs = {} if abi_kwargs is None else abi_kwargs
        for key in abi_kwargs:
            self._check_varname(key)
        args = list(abi_args)[:]
        args.extend(list(abi_kwargs.items()))
        self._vars = dict(args)
        self.set_structure(structure)
        if isinstance(pseudos, str):
            pseudos = [pseudos]
        if pseudo_dir is not None:
            pseudo_dir = os.path.abspath(pseudo_dir)
            if not os.path.isdir(pseudo_dir):
                raise self.Error(f'Directory {pseudo_dir} does not exist')
            pseudos = [os.path.join(pseudo_dir, p) for p in pseudos]
        try:
            self._pseudos = PseudoTable.as_table(pseudos).get_pseudos_for_structure(self.structure)
        except ValueError as exc:
            raise self.Error(str(exc))
        if comment is not None:
            self.set_comment(comment)

    def as_dict(self):
        """JSON interface used in pymatgen for easier serialization."""
        abi_args = []
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            abi_args.append((key, value))
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self.structure.as_dict(), 'pseudos': [p.as_dict() for p in self.pseudos], 'comment': self.comment, 'abi_args': abi_args}

    @property
    def vars(self):
        """Dictionary with variables."""
        return self._vars

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """JSON interface used in pymatgen for easier serialization."""
        pseudos = [Pseudo.from_file(p['filepath']) for p in dct['pseudos']]
        return cls(dct['structure'], pseudos, comment=dct['comment'], abi_args=dct['abi_args'])

    def add_abiobjects(self, *abi_objects):
        """
        This function receive a list of AbiVarable objects and add
        the corresponding variables to the input.
        """
        dct = {}
        for obj in abi_objects:
            if not hasattr(obj, 'to_abivars'):
                raise TypeError(f'type {type(obj).__name__} does not have `to_abivars` method')
            dct.update(self.set_vars(obj.to_abivars()))
        return dct

    def __setitem__(self, key, value):
        if key in _TOLVARS_SCF and hasattr(self, '_vars') and any((tol in self._vars and tol != key for tol in _TOLVARS_SCF)):
            logger.info(f'Replacing previously set tolerance variable: {self.remove_vars(_TOLVARS_SCF, strict=False)}.')
        return super().__setitem__(key, value)

    def _check_varname(self, key):
        if key in GEOVARS:
            raise self.Error('You cannot set the value of a variable associated to the structure.\nUse Structure objects to prepare the input file.')

    def to_str(self, post=None, with_structure=True, with_pseudos=True, exclude=None):
        """
        String representation.

        Args:
            post: String that will be appended to the name of the variables
                Note that post is usually autodetected when we have multiple datatasets
                It is mainly used when we have an input file with a single dataset
                so that we can prevent the code from adding "1" to the name of the variables
                (In this case, indeed, Abinit complains if ndtset=1 is not specified
                and we don't want ndtset=1 simply because the code will start to add
                _DS1_ to all the input and output files.
            with_structure: False if section with structure variables should not be printed.
            with_pseudos: False if JSON section with pseudo data should not be added.
            exclude: List of variable names that should be ignored.
        """
        lines = []
        if self.comment:
            lines.append('# ' + self.comment.replace('\n', '\n#'))
        post = post if post is not None else ''
        exclude = set(exclude) if exclude is not None else set()
        keys = sorted((k for k, v in self.items() if k not in exclude and v is not None))
        items = [(key, self[key]) for key in keys]
        if with_structure:
            items.extend(list(aobj.structure_to_abivars(self.structure).items()))
        for name, value in items:
            vname = name + post
            lines.append(str(InputVariable(vname, value)))
        out = '\n'.join(lines)
        if not with_pseudos:
            return out
        ppinfo = ['\n\n\n#<JSON>']
        psp_dict = {'pseudos': [p.as_dict() for p in self.pseudos]}
        ppinfo.extend(json.dumps(psp_dict, indent=4).splitlines())
        ppinfo.append('</JSON>')
        out += '\n#'.join(ppinfo)
        return out

    @property
    def comment(self):
        """Optional string with comment. None if comment is not set."""
        try:
            return self._comment
        except AttributeError:
            return None

    def set_comment(self, comment):
        """Set a comment to be included at the top of the file."""
        self._comment = comment

    @property
    def structure(self):
        """The |Structure| object associated to this input."""
        return self._structure

    def set_structure(self, structure: Structure):
        """Set structure."""
        self._structure = as_structure(structure)
        m = self.structure.lattice.matrix
        if np.dot(np.cross(m[0], m[1]), m[2]) <= 0:
            raise self.Error('The triple product of the lattice vector is negative. Use structure.abi_sanitize.')
        return self._structure

    def set_kmesh(self, ngkpt, shiftk, kptopt=1):
        """
        Set the variables for the sampling of the BZ.

        Args:
            ngkpt: Monkhorst-Pack divisions
            shiftk: List of shifts.
            kptopt: Option for the generation of the mesh.
        """
        shiftk = np.reshape(shiftk, (-1, 3))
        return self.set_vars(ngkpt=ngkpt, kptopt=kptopt, nshiftk=len(shiftk), shiftk=shiftk)

    def set_gamma_sampling(self):
        """Gamma-only sampling of the BZ."""
        return self.set_kmesh(ngkpt=(1, 1, 1), shiftk=(0, 0, 0))

    def set_kpath(self, ndivsm, kptbounds=None, iscf=-2):
        """
        Set the variables for the computation of the electronic band structure.

        Args:
            ndivsm: Number of divisions for the smallest segment.
            kptbounds: k-points defining the path in k-space.
                If None, we use the default high-symmetry k-path defined in the pymatgen database.
        """
        if kptbounds is None:
            hsym_kpath = HighSymmKpath(self.structure)
            name2frac_coords = hsym_kpath.kpath['kpoints']
            kpath = hsym_kpath.kpath['path']
            frac_coords, names = ([], [])
            for segment in kpath:
                for name in segment:
                    fc = name2frac_coords[name]
                    frac_coords.append(fc)
                    names.append(name)
            kptbounds = np.array(frac_coords)
        kptbounds = np.reshape(kptbounds, (-1, 3))
        return self.set_vars(kptbounds=kptbounds, kptopt=-(len(kptbounds) - 1), ndivsm=ndivsm, iscf=iscf)

    def set_spin_mode(self, spin_mode):
        """
        Set the variables used to the treat the spin degree of freedom.
        Return dictionary with the variables that have been removed.

        Args:
            spin_mode: SpinMode object or string. Possible values for string are:

            - polarized
            - unpolarized
            - afm (anti-ferromagnetic)
            - spinor (non-collinear magnetism)
            - spinor_nomag (non-collinear, no magnetism)
        """
        old_vars = self.pop_vars(['nsppol', 'nspden', 'nspinor'])
        self.add_abiobjects(aobj.SpinMode.as_spinmode(spin_mode))
        return old_vars

    @property
    def pseudos(self):
        """List of |Pseudo| objects."""
        return self._pseudos

    @property
    def ispaw(self):
        """True if PAW calculation."""
        return all((p.ispaw for p in self.pseudos))

    @property
    def isnc(self):
        """True if norm-conserving calculation."""
        return all((p.isnc for p in self.pseudos))

    def new_with_vars(self, *args, **kwargs):
        """
        Return a new input with the given variables.

        Example:
            new = input.new_with_vars(ecut=20)
        """
        new = self.deepcopy()
        new.set_vars(*args, **kwargs)
        return new

    def pop_tolerances(self):
        """
        Remove all the tolerance variables present in self.
        Return dictionary with the variables that have been removed.
        """
        return self.remove_vars(_TOLVARS, strict=False)

    def pop_irdvars(self):
        """
        Remove all the `ird*` variables present in self.
        Return dictionary with the variables that have been removed.
        """
        return self.remove_vars(_IRDVARS, strict=False)