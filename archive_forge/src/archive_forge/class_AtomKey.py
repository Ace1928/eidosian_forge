import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
class AtomKey:
    """Class for dict keys to reference atom coordinates.

    AtomKeys capture residue and disorder information together, and
    provide a no-whitespace string key for .pic files.

    Supports rich comparison and multiple ways to instantiate.

    AtomKeys contain:
     residue position (respos), insertion code (icode), 1 or 3 char residue
     name (resname), atom name (atm), altloc (altloc), and occupancy (occ)

    Use :data:`AtomKey.fields` to get the index to the component of interest by
    name:

    Get C-alpha atoms from IC_Chain atomArray and atomArrayIndex with
    AtomKeys::

        atmNameNdx = internal_coords.AtomKey.fields.atm
        CaSelection = [
            atomArrayIndex.get(k)
            for k in atomArrayIndex.keys()
            if k.akl[atmNameNdx] == "CA"
        ]
        AtomArrayCa = atomArray[CaSelection]

    Get all phenylalanine atoms in a chain::

        resNameNdx = internal_coords.AtomKey.fields.resname
        PheSelection = [
            atomArrayIndex.get(k)
            for k in atomArrayIndex.keys()
            if k.akl[resNameNdx] == "F"
        ]
        AtomArrayPhe = atomArray[PheSelection]

    'resname' will be the uppercase 1-letter amino acid code if one of the 20
    standard residues, otherwise the supplied 3-letter code.  Supplied as input
    or read from .rbase attribute of :class:`IC_Residue`.

    Attributes
    ----------
    akl: tuple
        All six fields of AtomKey
    fieldNames: tuple (Class Attribute)
        Mapping of key index positions to names
    fields: namedtuple (Class Attribute)
        Mapping of field names to index positions.
    id: str
        '_'-joined AtomKey fields, excluding 'None' fields
    atom_re: compiled regex (Class Attribute)
        A compiled regular expression matching the string form of the key
    d2h: bool (Class Attribute) default False
        Convert D atoms to H on input if True; must also modify
        :data:`IC_Residue.accept_atoms`
    missing: bool default False
        AtomKey __init__'d from string is probably missing, set this flag to
        note the issue.  Set by :meth:`.IC_Residue.rak`
    ric: IC_Residue default None
        *If* initialised with IC_Residue, this references the IC_residue

    Methods
    -------
    altloc_match(other)
        Returns True if this AtomKey matches other AtomKey excluding altloc
        and occupancy fields
    is_backbone()
        Returns True if atom is N, CA, C, O or H
    atm()
        Returns atom name, e.g. N, CA, CB, etc.
    cr_class()
        Returns covalent radii class e.g. Csb

    """
    atom_re = re.compile('^(?P<respos>-?\\d+)(?P<icode>[A-Za-z])?_(?P<resname>[a-zA-Z]+)_(?P<atm>[A-Za-z0-9]+)(?:_(?P<altloc>\\w))?(?:_(?P<occ>-?\\d\\.\\d+?))?$')
    'Pre-compiled regular expression to match an AtomKey string.'
    _endnum_re = re.compile('\\D+(\\d+)$')
    fieldNames = ('respos', 'icode', 'resname', 'atm', 'altloc', 'occ')
    _fieldsDef = namedtuple('_fieldsDef', ['respos', 'icode', 'resname', 'atm', 'altloc', 'occ'])
    fields = _fieldsDef(0, 1, 2, 3, 4, 5)
    'Use this namedtuple to access AtomKey fields.  See :class:`AtomKey`'
    d2h = False
    'Set True to convert D Deuterium to H Hydrogen on input.'

    def __init__(self, *args: Union[IC_Residue, Atom, List, Dict, str], **kwargs: str) -> None:
        """Initialize AtomKey with residue and atom data.

        Examples of acceptable input::

            (<IC_Residue>, 'CA', ...)    : IC_Residue with atom info
            (<IC_Residue>, <Atom>)       : IC_Residue with Biopython Atom
            ([52, None, 'G', 'CA', ...])  : list of ordered data fields
            (52, None, 'G', 'CA', ...)    : multiple ordered arguments
            ({respos: 52, icode: None, atm: 'CA', ...}) : dict with fieldNames
            (respos: 52, icode: None, atm: 'CA', ...) : kwargs with fieldNames
            52_G_CA, 52B_G_CA, 52_G_CA_0.33, 52_G_CA_B_0.33  : id strings
        """
        akl: List[Optional[str]] = []
        self.ric = None
        for arg in args:
            if isinstance(arg, str):
                if '_' in arg:
                    m = self.atom_re.match(arg)
                    if m is not None:
                        if akl != []:
                            raise Exception('Atom Key init full key not first argument: ' + arg)
                        akl = list(map(m.group, AtomKey.fieldNames))
                else:
                    akl.append(arg)
            elif isinstance(arg, IC_Residue):
                if akl != []:
                    raise Exception('Atom Key init Residue not first argument')
                akl = list(arg.rbase)
                self.ric = arg
            elif isinstance(arg, Atom):
                if 3 != len(akl):
                    raise Exception('Atom Key init Atom before Residue info')
                akl.append(arg.name)
                if not IC_Residue.no_altloc:
                    altloc = arg.altloc
                    akl.append(altloc if altloc != ' ' else None)
                    occ = float(arg.occupancy)
                    akl.append(str(occ) if occ != 1.0 else None)
                else:
                    akl += [None, None]
            elif isinstance(arg, (list, tuple)):
                akl += arg
            elif isinstance(arg, dict):
                for k in AtomKey.fieldNames:
                    akl.append(arg.get(k, None))
            else:
                raise Exception('Atom Key init not recognised')
        for i in range(len(akl), 6):
            if len(akl) <= i:
                fld = kwargs.get(AtomKey.fieldNames[i])
                if fld is not None:
                    akl.append(fld)
        if isinstance(akl[0], Integral):
            akl[0] = str(akl[0])
        if self.d2h:
            atmNdx = AtomKey.fields.atm
            if akl[atmNdx][0] == 'D':
                akl[atmNdx] = re.sub('D', 'H', akl[atmNdx], count=1)
        self.id = '_'.join([''.join(filter(None, akl[:2])), str(akl[2]), '_'.join(filter(None, akl[3:]))])
        akl += [None] * (6 - len(akl))
        self.akl = tuple(akl)
        self._hash = hash(self.akl)
        self.missing = False

    def __deepcopy__(self, memo):
        """Deep copy implementation for AtomKey."""
        existing = memo.get(id(self), False)
        if existing:
            return existing
        dup = type(self).__new__(self.__class__)
        memo[id(self)] = dup
        dup.__dict__.update(self.__dict__)
        if self.ric is not None:
            dup.ric = memo[id(self.ric)]
        return dup

    def __repr__(self) -> str:
        """Repr string from id."""
        return self.id

    def __hash__(self) -> int:
        """Hash calculated at init from akl tuple."""
        return self._hash
    _backbone_sort_keys = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    _sidechain_sort_keys = {'CB': 1, 'CG': 2, 'CG1': 2, 'OG': 2, 'OG1': 2, 'SG': 2, 'CG2': 3, 'CD': 4, 'CD1': 4, 'SD': 4, 'OD1': 4, 'ND1': 4, 'CD2': 5, 'ND2': 5, 'OD2': 5, 'CE': 6, 'NE': 6, 'CE1': 6, 'OE1': 6, 'NE1': 6, 'CE2': 7, 'OE2': 7, 'NE2': 7, 'CE3': 8, 'CZ': 9, 'CZ2': 9, 'NZ': 9, 'NH1': 10, 'OH': 10, 'CZ3': 10, 'CH2': 11, 'NH2': 11, 'OXT': 12, 'H': 13}
    _greek_sort_keys = {'A': 0, 'B': 1, 'G': 2, 'D': 3, 'E': 4, 'Z': 5, 'H': 6}

    def altloc_match(self, other: 'AtomKey') -> bool:
        """Test AtomKey match to other discounting occupancy and altloc."""
        if isinstance(other, type(self)):
            return self.akl[:4] == other.akl[:4]
        else:
            return NotImplemented

    def is_backbone(self) -> bool:
        """Return True if is N, C, CA, O, or H."""
        return self.akl[self.fields.atm] in ('N', 'C', 'CA', 'O', 'H')

    def atm(self) -> str:
        """Return atom name : N, CA, CB, O etc."""
        return self.akl[self.fields.atm]

    def cr_class(self) -> Union[str, None]:
        """Return covalent radii class for atom or None."""
        akl = self.akl
        atmNdx = self.fields.atm
        try:
            return residue_atom_bond_state['X'][akl[atmNdx]]
        except KeyError:
            try:
                resNdx = self.fields.resname
                return residue_atom_bond_state[akl[resNdx]][akl[atmNdx]]
            except KeyError:
                return 'Hsb' if akl[atmNdx][0] == 'H' else None

    def _cmp(self, other: 'AtomKey') -> Tuple[int, int]:
        """Comparison function ranking self vs. other.

        Priority is lower value, i.e. (CA, CB) gives (0, 1) for sorting.
        """
        for i in range(6):
            s, o = (self.akl[i], other.akl[i])
            if s != o:
                if s is None and o is not None:
                    return (0, 1)
                elif o is None and s is not None:
                    return (1, 0)
                if AtomKey.fields.atm != i:
                    if AtomKey.fields.occ == i:
                        oi = int(float(s) * 100)
                        si = int(float(o) * 100)
                        return (si, oi)
                    elif AtomKey.fields.respos == i:
                        return (int(s), int(o))
                    elif AtomKey.fields.resname == i:
                        sac, oac = (self.akl[AtomKey.fields.altloc], other.akl[AtomKey.fields.altloc])
                        if sac is not None:
                            if oac is not None:
                                return (ord(sac), ord(oac))
                            else:
                                return (1, 0)
                        elif oac is not None:
                            return (0, 1)
                    return (ord(s), ord(o))
                sb = self._backbone_sort_keys.get(s, None)
                ob = self._backbone_sort_keys.get(o, None)
                if sb is not None and ob is not None:
                    return (sb, ob)
                elif sb is not None and ob is None:
                    return (0, 1)
                elif sb is None and ob is not None:
                    return (1, 0)
                ss = self._sidechain_sort_keys.get(s, None)
                os = self._sidechain_sort_keys.get(o, None)
                if ss is not None and os is not None:
                    return (ss, os)
                elif ss is not None and os is None:
                    return (0, 1)
                elif ss is None and os is not None:
                    return (1, 0)
                s0, s1, o0, o1 = (s[0], s[1], o[0], o[1])
                s1d, o1d = (s1.isdigit(), o1.isdigit())
                if 'H' == s0 and 'H' == o0:
                    if s1 == o1 or (s1d and o1d):
                        enmS = self._endnum_re.findall(s)
                        enmO = self._endnum_re.findall(o)
                        if enmS != [] and enmO != []:
                            return (int(enmS[0]), int(enmO[0]))
                        elif enmS == []:
                            return (0, 1)
                        else:
                            return (1, 0)
                    elif s1d:
                        return (0, 1)
                    elif o1d:
                        return (1, 0)
                    else:
                        return (self._greek_sort_keys[s1], self._greek_sort_keys[o1])
                return (int(s), int(o))
        return (1, 1)

    def __ne__(self, other: object) -> bool:
        """Test for inequality."""
        if isinstance(other, type(self)):
            return self.akl != other.akl
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Test for equality."""
        if isinstance(other, type(self)):
            return self.akl == other.akl
        else:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Test greater than."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] > rslt[1]
        else:
            return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Test greater or equal."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] >= rslt[1]
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Test less than."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] < rslt[1]
        else:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        """Test less or equal."""
        if isinstance(other, type(self)):
            rslt = self._cmp(other)
            return rslt[0] <= rslt[1]
        else:
            return NotImplemented