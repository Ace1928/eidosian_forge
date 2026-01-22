from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class Pseudo(MSONable, abc.ABC):
    """
    Abstract base class defining the methods that must be
    implemented by the concrete pseudo-potential sub-classes.
    """

    @classmethod
    def as_pseudo(cls, obj):
        """
        Convert obj into a pseudo. Accepts:

            * Pseudo object.
            * string defining a valid path.
        """
        return obj if isinstance(obj, cls) else cls.from_file(obj)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Build an instance of a concrete Pseudo subclass from filename.
        Note: the parser knows the concrete class that should be instantiated
        Client code should rely on the abstract interface provided by Pseudo.
        """
        return PseudoParser().parse(filename)

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('md5', 'Z', 'Z_val', 'l_max')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs)) and self.__class__ == other.__class__

    def __repr__(self) -> str:
        try:
            return f'<{type(self).__name__} at {os.path.relpath(self.filepath)}>'
        except Exception:
            return f'<{type(self).__name__} at {self.filepath}>'

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self, verbose=0) -> str:
        """String representation."""
        lines: list[str] = []
        lines.append(f'<{type(self).__name__}: {self.basename}>')
        lines.append('  summary: ' + self.summary.strip())
        lines.append(f'  number of valence electrons: {self.Z_val}')
        lines.append(f'  maximum angular momentum: {l2str(self.l_max)}')
        lines.append(f'  angular momentum for local part: {l2str(self.l_local)}')
        lines.append(f'  XC correlation: {self.xc}')
        lines.append(f'  supports spin-orbit: {self.supports_soc}')
        if self.isnc:
            lines.append(f'  radius for non-linear core correction: {self.nlcc_radius}')
        if self.has_hints:
            for accuracy in ('low', 'normal', 'high'):
                hint = self.hint_for_accuracy(accuracy=accuracy)
                lines.append(f'  hint for {accuracy} accuracy: {hint}')
        return '\n'.join(lines)

    @property
    @abc.abstractmethod
    def summary(self) -> str:
        """String summarizing the most important properties."""

    @property
    def filepath(self) -> str:
        """Absolute path to pseudopotential file."""
        return os.path.abspath(self.path)

    @property
    def basename(self) -> str:
        """File basename."""
        return os.path.basename(self.filepath)

    @property
    @abc.abstractmethod
    def Z(self) -> int:
        """The atomic number of the atom."""

    @property
    @abc.abstractmethod
    def Z_val(self) -> int:
        """Valence charge."""

    @property
    def type(self) -> str:
        """Type of pseudo."""
        return type(self).__name__

    @property
    def element(self) -> Element:
        """Pymatgen Element."""
        try:
            return Element.from_Z(self.Z)
        except (KeyError, IndexError):
            return Element.from_Z(int(self.Z))

    @property
    def symbol(self) -> str:
        """Element symbol."""
        return self.element.symbol

    @property
    @abc.abstractmethod
    def l_max(self) -> int:
        """Maximum angular momentum."""

    @property
    @abc.abstractmethod
    def l_local(self) -> int:
        """Angular momentum used for the local part."""

    @property
    def isnc(self) -> bool:
        """True if norm-conserving pseudopotential."""
        return isinstance(self, NcPseudo)

    @property
    def ispaw(self) -> bool:
        """True if PAW pseudopotential."""
        return isinstance(self, PawPseudo)

    @lazy_property
    def md5(self):
        """MD5 hash value."""
        return self.compute_md5()

    def compute_md5(self):
        """Compute and return MD5 hash value."""
        with open(self.path) as file:
            text = file.read()
            md5 = hashlib.new('md5', usedforsecurity=False)
            md5.update(text.encode('utf-8'))
            return md5.hexdigest()

    @property
    @abc.abstractmethod
    def supports_soc(self):
        """
        True if the pseudo can be used in a calculation with spin-orbit coupling.
        Base classes should provide a concrete implementation that computes this value.
        """

    def as_dict(self, **kwargs):
        """Return dictionary for MSONable protocol."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'basename': self.basename, 'type': self.type, 'symbol': self.symbol, 'Z': self.Z, 'Z_val': self.Z_val, 'l_max': self.l_max, 'md5': self.md5, 'filepath': self.filepath}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build instance from dictionary (MSONable protocol)."""
        new = cls.from_file(dct['filepath'])
        if dct.get('md5') != new.md5:
            raise ValueError(f'The md5 found in file does not agree with the one in dict\nReceived {dct['md5']}\nComputed {new.md5}')
        return new

    def as_tmpfile(self, tmpdir=None):
        """
        Copy the pseudopotential to a temporary a file and returns a new pseudopotential object.
        Useful for unit tests in which we have to change the content of the file.

        Args:
            tmpdir: If None, a new temporary directory is created and files are copied here
                else tmpdir is used.
        """
        tmpdir = tempfile.mkdtemp() if tmpdir is None else tmpdir
        new_path = os.path.join(tmpdir, self.basename)
        shutil.copy(self.filepath, new_path)
        root, _ext = os.path.splitext(self.filepath)
        dj_report = root + '.djrepo'
        if os.path.isfile(dj_report):
            shutil.copy(dj_report, os.path.join(tmpdir, os.path.basename(dj_report)))
        new = type(self).from_file(new_path)
        if self.has_dojo_report:
            new.dojo_report = self.dojo_report.deepcopy()
        return new

    @property
    def has_dojo_report(self):
        """True if the pseudo has an associated `DOJO_REPORT` section."""
        return hasattr(self, 'dojo_report') and self.dojo_report

    @property
    def djrepo_path(self):
        """The path of the djrepo file. None if file does not exist."""
        root, _ext = os.path.splitext(self.filepath)
        return root + '.djrepo'

    def hint_for_accuracy(self, accuracy='normal'):
        """
        Returns a Hint object with the suggested value of ecut [Ha] and
        pawecutdg [Ha] for the given accuracy.
        ecut and pawecutdg are set to zero if no hint is available.

        Args:
            accuracy: ["low", "normal", "high"]
        """
        if not self.has_dojo_report:
            return Hint(ecut=0.0, pawecutdg=0.0)
        if 'hints' in self.dojo_report:
            return Hint.from_dict(self.dojo_report['hints'][accuracy])
        if 'ppgen_hints' in self.dojo_report:
            return Hint.from_dict(self.dojo_report['ppgen_hints'][accuracy])
        return Hint(ecut=0.0, pawecutdg=0.0)

    @property
    def has_hints(self):
        """True if self provides hints on the cutoff energy."""
        for acc in ['low', 'normal', 'high']:
            try:
                if self.hint_for_accuracy(acc) is None:
                    return False
            except KeyError:
                return False
        return True

    def open_pspsfile(self, ecut=20, pawecutdg=None):
        """
        Calls Abinit to compute the internal tables for the application of the
        pseudopotential part. Returns PspsFile object providing methods
        to plot and analyze the data or None if file is not found or it's not readable.

        Args:
            ecut: Cutoff energy in Hartree.
            pawecutdg: Cutoff energy for the PAW double grid.
        """
        from abipy.abio.factories import gs_input
        from abipy.core.structure import Structure
        from abipy.electrons.psps import PspsFile
        from abipy.flowtk import AbinitTask
        lattice = 10 * np.eye(3)
        structure = Structure(lattice, [self.element], coords=[[0, 0, 0]])
        if self.ispaw and pawecutdg is None:
            pawecutdg = ecut * 4
        inp = gs_input(structure, pseudos=[self], ecut=ecut, pawecutdg=pawecutdg, spin_mode='unpolarized', kppa=1)
        inp['prtpsps'] = -1
        task = AbinitTask.temp_shell_task(inp)
        task.start_and_wait()
        filepath = task.outdir.has_abiext('_PSPS.nc')
        if not filepath:
            logger.critical(f'Cannot find PSPS.nc file in {task.outdir}')
            return None
        try:
            return PspsFile(filepath)
        except Exception as exc:
            logger.critical(f'Exception while reading PSPS file at {filepath}:\n{exc}')
            return None