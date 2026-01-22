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
class PseudoTable(collections.abc.Sequence, MSONable):
    """
    Define the pseudopotentials from the element table.
    Individidual elements are accessed by name, symbol or atomic number.

    For example, the following all retrieve iron:

    print elements[26]
    Fe
    print elements.Fe
    Fe
    print elements.symbol('Fe')
    Fe
    print elements.name('iron')
    Fe
    print elements.isotope('Fe')
    Fe
    """

    @classmethod
    def as_table(cls, items):
        """Return an instance of PseudoTable from the iterable items."""
        if isinstance(items, cls):
            return items
        return cls(items)

    @classmethod
    def from_dir(cls, top, exts=None, exclude_dirs='_*') -> Self | None:
        """
        Find all pseudos in the directory tree starting from top.

        Args:
            top: Top of the directory tree
            exts: List of files extensions. if exts == "all_files"
                    we try to open all files in top
            exclude_dirs: Wildcard used to exclude directories.

        Returns:
            PseudoTable sorted by atomic number Z.
        """
        pseudos = []
        if exts == 'all_files':
            for filepath in [os.path.join(top, fn) for fn in os.listdir(top)]:
                if os.path.isfile(filepath):
                    try:
                        pseudo = Pseudo.from_file(filepath)
                        if pseudo:
                            pseudos.append(pseudo)
                        else:
                            logger.info(f'Skipping file {filepath}')
                    except Exception:
                        logger.info(f'Skipping file {filepath}')
            if not pseudos:
                logger.warning(f'No pseudopotentials parsed from folder {top}')
                return None
            logger.info(f'Creating PseudoTable with {len(pseudos)} pseudopotentials')
        else:
            if exts is None:
                exts = ('psp8',)
            for pseudo in find_exts(top, exts, exclude_dirs=exclude_dirs):
                try:
                    pseudos.append(Pseudo.from_file(pseudo))
                except Exception as exc:
                    logger.critical(f'Error in {pseudo}:\n{exc}')
        return cls(pseudos).sort_by_z()

    def __init__(self, pseudos: Sequence[Pseudo]) -> None:
        """
        Args:
            pseudos: List of pseudopotentials or filepaths.
        """
        if not isinstance(pseudos, collections.abc.Iterable):
            pseudos = [pseudos]
        if isinstance(pseudos, str):
            pseudos = [pseudos]
        self._pseudos_with_z = defaultdict(list)
        for pseudo in pseudos:
            if not isinstance(pseudo, Pseudo):
                pseudo = Pseudo.from_file(pseudo)
            if pseudo is not None:
                self._pseudos_with_z[pseudo.Z].append(pseudo)
        for z in self.zlist:
            pseudo_list = self._pseudos_with_z[z]
            symbols = [p.symbol for p in pseudo_list]
            symbol = symbols[0]
            if any((symb != symbol for symb in symbols)):
                raise ValueError(f'All symbols must be equal while they are: {symbols}')
            setattr(self, symbol, pseudo_list)

    def __getitem__(self, Z):
        """Retrieve pseudos for the atomic number z. Accepts both int and slice objects."""
        if isinstance(Z, slice):
            assert Z.stop is not None
            pseudos = []
            for znum in iterator_from_slice(Z):
                pseudos.extend(self._pseudos_with_z[znum])
            return type(self)(pseudos)
        return type(self)(self._pseudos_with_z[Z])

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __iter__(self) -> Iterator[Pseudo]:
        """Process the elements in Z order."""
        for z in self.zlist:
            yield from self._pseudos_with_z[z]

    def __repr__(self) -> str:
        return f'<{type(self).__name__} at {id(self)}>'

    def __str__(self) -> str:
        return self.to_table()

    @property
    def allnc(self) -> bool:
        """True if all pseudos are norm-conserving."""
        return all((p.isnc for p in self))

    @property
    def allpaw(self):
        """True if all pseudos are PAW."""
        return all((p.ispaw for p in self))

    @property
    def zlist(self):
        """Ordered list with the atomic numbers available in the table."""
        return sorted(self._pseudos_with_z)

    def as_dict(self, **kwargs):
        """Return dictionary for MSONable protocol."""
        dct = {}
        for p in self:
            k, count = (p.element.name, 1)
            while k in dct:
                k += f'{k.split('#')[0]}#{count}'
                count += 1
            dct.update({k: p.as_dict()})
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Build instance from dictionary (MSONable protocol)."""
        pseudos = []
        for k, v in dct.items():
            if not k.startswith('@'):
                pseudos.append(MontyDecoder().process_decoded(v))
        return cls(pseudos)

    def is_complete(self, zmax=118) -> bool:
        """True if table is complete i.e. all elements with Z < zmax have at least on pseudopotential."""
        return all((self[z] for z in range(1, zmax)))

    def all_combinations_for_elements(self, element_symbols):
        """
        Return a list with all the possible combination of pseudos
        for the given list of element_symbols.
        Each item is a list of pseudopotential objects.

        Example:
            table.all_combinations_for_elements(["Li", "F"])
        """
        dct = {}
        for symbol in element_symbols:
            dct[symbol] = self.select_symbols(symbol, ret_list=True)
        from itertools import product
        return list(product(*dct.values()))

    def pseudo_with_symbol(self, symbol, allow_multi=False):
        """
        Return the pseudo with the given chemical symbol.

        Args:
            symbols: String with the chemical symbol of the element
            allow_multi: By default, the method raises ValueError
                if multiple occurrences are found. Use allow_multi to prevent this.

        Raises:
            ValueError if symbol is not found or multiple occurrences are present and not allow_multi
        """
        pseudos = self.select_symbols(symbol, ret_list=True)
        if not pseudos or (len(pseudos) > 1 and (not allow_multi)):
            raise ValueError(f'Found {len(pseudos)} occurrences of symbol={symbol!r}')
        if not allow_multi:
            return pseudos[0]
        return pseudos

    def pseudos_with_symbols(self, symbols):
        """
        Return the pseudos with the given chemical symbols.

        Raises:
            ValueError if one of the symbols is not found or multiple occurrences are present.
        """
        pseudos = self.select_symbols(symbols, ret_list=True)
        found_symbols = [p.symbol for p in pseudos]
        duplicated_elements = [s for s, o in collections.Counter(found_symbols).items() if o > 1]
        if duplicated_elements:
            raise ValueError(f'Found multiple occurrences of symbol(s) {', '.join(duplicated_elements)}')
        missing_symbols = [s for s in symbols if s not in found_symbols]
        if missing_symbols:
            raise ValueError(f'Missing data for symbol(s) {', '.join(missing_symbols)}')
        return pseudos

    def select_symbols(self, symbols, ret_list=False):
        """
        Return a PseudoTable with the pseudopotentials with the given list of chemical symbols.

        Args:
            symbols: str or list of symbols
                Prepend the symbol string with "-", to exclude pseudos.
            ret_list: if True a list of pseudos is returned instead of a PseudoTable
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        exclude = symbols[0].startswith('-')
        if exclude:
            if not all((s.startswith('-') for s in symbols)):
                raise ValueError('When excluding symbols, all strings must start with `-`')
            symbols = [s[1:] for s in symbols]
        symbols = set(symbols)
        pseudos = []
        for p in self:
            if exclude:
                if p.symbol in symbols:
                    continue
            elif p.symbol not in symbols:
                continue
            pseudos.append(p)
        if ret_list:
            return pseudos
        return type(self)(pseudos)

    def get_pseudos_for_structure(self, structure: Structure):
        """
        Return the list of Pseudo objects to be used for this Structure.

        Args:
            structure: pymatgen Structure.

        Raises:
            `ValueError` if one of the chemical symbols is not found or
            multiple occurrences are present in the table.
        """
        return self.pseudos_with_symbols(structure.symbol_set)

    def print_table(self, stream=sys.stdout, filter_function=None):
        """
        A pretty ASCII printer for the periodic table, based on some filter_function.

        Args:
            stream: file-like object
            filter_function:
                A filtering function that take a Pseudo as input and returns a boolean.
                For example, setting filter_function = lambda p: p.Z_val > 2 will print
                a periodic table containing only pseudos with Z_val > 2.
        """
        print(self.to_table(filter_function=filter_function), file=stream)

    def to_table(self, filter_function=None):
        """Return string with data in tabular form."""
        table = []
        for p in self:
            if filter_function is not None and filter_function(p):
                continue
            table.append([p.basename, p.symbol, p.Z_val, p.l_max, p.l_local, p.xc, p.type])
        return tabulate(table, headers=['basename', 'symbol', 'Z_val', 'l_max', 'l_local', 'XC', 'type'], tablefmt='grid')

    def sorted(self, attrname, reverse=False):
        """
        Sort the table according to the value of attribute attrname.

        Returns:
            New class: `PseudoTable` object
        """
        attrs = []
        for i, pseudo in self:
            try:
                a = getattr(pseudo, attrname)
            except AttributeError:
                a = np.inf
            attrs.append((i, a))
        return type(self)([self[a[0]] for a in sorted(attrs, key=lambda t: t[1], reverse=reverse)])

    def sort_by_z(self):
        """Return a new PseudoTable with pseudos sorted by Z."""
        return type(self)(sorted(self, key=lambda p: p.Z))

    def select(self, condition) -> PseudoTable:
        """Select only those pseudopotentials for which condition is True.

        Args:
            condition:
                Function that accepts a Pseudo object and returns True or False.

        Returns:
            PseudoTable: New PseudoTable instance with pseudos for which condition is True.
        """
        return type(self)([p for p in self if condition(p)])

    def with_dojo_report(self):
        """Select pseudos containing the DOJO_REPORT section. Return new class:`PseudoTable` object."""
        return self.select(condition=lambda p: p.has_dojo_report)

    def select_rows(self, rows):
        """
        Return new class:`PseudoTable` object with pseudos in the given rows of the periodic table.
        rows can be either a int or a list of integers.
        """
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        return type(self)([p for p in self if p.element.row in rows])

    def select_family(self, family):
        """Return PseudoTable with element belonging to the specified family, e.g. family="alkaline"."""
        return type(self)([p for p in self if getattr(p.element, 'is_' + family)])