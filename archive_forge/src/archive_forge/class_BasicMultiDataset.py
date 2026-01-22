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
class BasicMultiDataset:
    """
    This object is essentially a list of BasicAbinitInput objects.
    that provides an easy-to-use interface to apply global changes to the
    the inputs stored in the objects.

    Let's assume for example that multi contains two BasicAbinitInput objects and we
    want to set `ecut` to 1 in both dictionaries. The direct approach would be:

        for inp in multi:
            inp.set_vars(ecut=1)

    or alternatively:

        for i in range(multi.ndtset):
            multi[i].set_vars(ecut=1)

    BasicMultiDataset provides its own implementation of __getattr__ so that one can simply use:

        multi.set_vars(ecut=1)

        multi.get("ecut") returns a list of values. It's equivalent to:

            [inp["ecut"] for inp in multi]

        Note that if "ecut" is not present in one of the input of multi, the corresponding entry is set to None.
        A default value can be specified with:

            multi.get("paral_kgb", 0)

    Warning:
        BasicMultiDataset does not support calculations done with different sets of pseudopotentials.
        The inputs can have different crystalline structures (as long as the atom types are equal)
        but each input in BasicMultiDataset must have the same set of pseudopotentials.
    """
    Error = BasicAbinitInputError

    def __init__(self, structure: Structure | Sequence[Structure], pseudos, pseudo_dir='', ndtset=1):
        """
        Args:
            structure: file with the structure, |Structure| object or dictionary with ABINIT geo variable
                Accepts also list of objects that can be converted to Structure object.
                In this case, however, ndtset must be equal to the length of the list.
            pseudos: String or list of string with the name of the pseudopotential files.
            pseudo_dir: Name of the directory where the pseudopotential files are located.
            ndtset: Number of datasets.
        """
        if isinstance(pseudos, Pseudo):
            pseudos = [pseudos]
        elif all((isinstance(p, Pseudo) for p in pseudos)):
            pseudos = PseudoTable(pseudos)
        else:
            if isinstance(pseudos, str):
                pseudos = [pseudos]
            pseudo_dir = os.path.abspath(pseudo_dir)
            pseudo_paths = [os.path.join(pseudo_dir, p) for p in pseudos]
            missing = [p for p in pseudo_paths if not os.path.isfile(p)]
            if missing:
                raise self.Error(f'Cannot find the following pseudopotential files:\n{missing}')
            pseudos = PseudoTable(pseudo_paths)
        if ndtset <= 0:
            raise ValueError(f'ndtset={ndtset!r} cannot be <=0')
        if not isinstance(structure, (list, tuple)):
            self._inputs = [BasicAbinitInput(structure=structure, pseudos=pseudos) for i in range(ndtset)]
        else:
            assert len(structure) == ndtset
            self._inputs = [BasicAbinitInput(structure=s, pseudos=pseudos) for s in structure]

    @classmethod
    def from_inputs(cls, inputs: list[BasicAbinitInput]) -> Self:
        """Build object from a list of BasicAbinitInput objects."""
        for inp in inputs:
            if any((p1 != p2 for p1, p2 in zip(inputs[0].pseudos, inp.pseudos))):
                raise ValueError('Pseudos must be consistent when from_inputs is invoked.')
        multi = cls(structure=[inp.structure for inp in inputs], pseudos=inputs[0].pseudos, ndtset=len(inputs))
        for inp, new_inp in zip(inputs, multi):
            new_inp.set_vars(**inp)
        return multi

    @classmethod
    def replicate_input(cls, input, ndtset):
        """Construct a multidataset with ndtset from the BasicAbinitInput input."""
        multi = cls(input.structure, input.pseudos, ndtset=ndtset)
        for inp in multi:
            inp.set_vars(**input)
        return multi

    @property
    def ndtset(self):
        """Number of inputs in self."""
        return len(self)

    @property
    def pseudos(self):
        """Pseudopotential objects."""
        return self[0].pseudos

    @property
    def ispaw(self):
        """True if PAW calculation."""
        return all((p.ispaw for p in self.pseudos))

    @property
    def isnc(self):
        """True if norm-conserving calculation."""
        return all((p.isnc for p in self.pseudos))

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, key):
        return self._inputs[key]

    def __iter__(self):
        return iter(self._inputs)

    def __getattr__(self, name):
        _inputs = self.__getattribute__('_inputs')
        m = getattr(_inputs[0], name)
        if m is None:
            raise AttributeError(f'Cannot find attribute {type(self).__name__}. Tried in {name} and then in BasicAbinitInput object')
        isattr = not callable(m)

        def on_all(*args, **kwargs):
            results = []
            for obj in self._inputs:
                a = getattr(obj, name)
                if callable(a):
                    results.append(a(*args, **kwargs))
                else:
                    results.append(a)
            return results
        if isattr:
            on_all = on_all()
        return on_all

    def __add__(self, other):
        """Self + other."""
        if isinstance(other, BasicAbinitInput):
            new_mds = BasicMultiDataset.from_inputs(self)
            new_mds.append(other)
            return new_mds
        if isinstance(other, BasicMultiDataset):
            new_mds = BasicMultiDataset.from_inputs(self)
            new_mds.extend(other)
            return new_mds
        raise NotImplementedError('Operation not supported')

    def __radd__(self, other):
        if isinstance(other, BasicAbinitInput):
            new_mds = BasicMultiDataset.from_inputs([other])
            new_mds.extend(self)
        elif isinstance(other, BasicMultiDataset):
            new_mds = BasicMultiDataset.from_inputs(other)
            new_mds.extend(self)
        else:
            raise NotImplementedError('Operation not supported')

    def append(self, abinit_input):
        """Add a |BasicAbinitInput| to the list."""
        assert isinstance(abinit_input, BasicAbinitInput)
        if any((p1 != p2 for p1, p2 in zip(abinit_input.pseudos, abinit_input.pseudos))):
            raise ValueError('Pseudos must be consistent when from_inputs is invoked.')
        self._inputs.append(abinit_input)

    def extend(self, abinit_inputs):
        """Extends self with a list of |BasicAbinitInput| objects."""
        assert all((isinstance(inp, BasicAbinitInput) for inp in abinit_inputs))
        for inp in abinit_inputs:
            if any((p1 != p2 for p1, p2 in zip(self[0].pseudos, inp.pseudos))):
                raise ValueError('Pseudos must be consistent when from_inputs is invoked.')
        self._inputs.extend(abinit_inputs)

    def addnew_from(self, dtindex):
        """Add a new entry in the multidataset by copying the input with index dtindex."""
        self.append(self[dtindex].deepcopy())

    def split_datasets(self):
        """Return list of |BasicAbinitInput| objects.."""
        return self._inputs

    def deepcopy(self):
        """Deep copy of the BasicMultiDataset."""
        return copy.deepcopy(self)

    @property
    def has_same_structures(self):
        """True if all inputs in BasicMultiDataset are equal."""
        return all((self[0].structure == inp.structure for inp in self))

    def __str__(self):
        return self.to_str()

    def to_str(self, with_pseudos=True):
        """
        String representation i.e. the input file read by Abinit.

        Args:
            with_pseudos: False if JSON section with pseudo data should not be added.
        """
        if self.ndtset > 1:
            lines = [f'ndtset {int(self.ndtset)}']

            def has_same_variable(kref, vref, other_inp):
                """True if variable kref is present in other_inp with the same value."""
                if kref not in other_inp:
                    return False
                otherv = other_inp[kref]
                return np.array_equal(vref, otherv)
            global_vars = set()
            for k0, v0 in self[0].items():
                isame = True
                for i in range(1, self.ndtset):
                    isame = has_same_variable(k0, v0, self[i])
                    if not isame:
                        break
                if isame:
                    global_vars.add(k0)
            w = 92
            if global_vars:
                lines.extend((w * '#', '### Global Variables.', w * '#'))
                for key in global_vars:
                    vname = key
                    lines.append(str(InputVariable(vname, self[0][key])))
            has_same_structures = self.has_same_structures
            if has_same_structures:
                lines.extend((w * '#', '#' + 'STRUCTURE'.center(w - 1), w * '#'))
                for key, value in aobj.structure_to_abivars(self[0].structure).items():
                    vname = key
                    lines.append(str(InputVariable(vname, value)))
            for i, inp in enumerate(self):
                header = f'### DATASET {i + 1} ###'
                is_last = i == self.ndtset - 1
                s = inp.to_str(post=str(i + 1), with_pseudos=is_last and with_pseudos, with_structure=not has_same_structures, exclude=global_vars)
                if s:
                    s = f'\n{len(header) * '#'}\n{header}\n{len(header) * '#'}\n{s}\n'
                lines.append(s)
            return '\n'.join(lines)
        return self[0].to_str(with_pseudos=with_pseudos)

    def write(self, filepath='run.abi'):
        """
        Write ndset input files to disk. The name of the file
        is constructed from the dataset index e.g. run0.abi.
        """
        root, ext = os.path.splitext(filepath)
        for i, inp in enumerate(self):
            p = f'{root}DS{i}' + ext
            inp.write(filepath=p)