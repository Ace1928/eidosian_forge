from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
class AdfKey(MSONable):
    """
    The basic input unit for ADF. A key is a string of characters that does not
    contain a delimiter (blank, comma or equal sign). A key may have multiple
    subkeys and a set of options.
    """
    block_keys = ('SCF', 'GEOMETRY', 'XC', 'UNITS', 'ATOMS', 'CHARGE', 'BASIS', 'SYMMETRY', 'RELATIVISTIC', 'OCCUPATIONS', 'SAVE', 'A1FIT', 'INTEGRATION', 'UNRESTRICTED', 'ZLMFIT', 'TITLE', 'EXACTDENSITY', 'TOTALENERGY', 'ANALYTICALFREQ')
    sub_keys = ('AtomDepQuality',)
    _full_blocks = ('GEOMETRY', 'SCF', 'UNITS', 'BASIS', 'ANALYTICALFREQ')

    def __init__(self, name, options=None, subkeys=None):
        """
        Initialization method.

        Args:
            name (str): The name of this key.
            options : Sized
                The options for this key. Each element can be a primitive object or
                a tuple/list with two elements: the first is the name and the second is a primitive object.
            subkeys (Sized): The subkeys for this key.

        Raises:
            ValueError: If elements in ``subkeys`` are not ``AdfKey`` objects.
        """
        self.name = name
        self.options = options if options is not None else []
        self.subkeys = subkeys if subkeys is not None else []
        if len(self.subkeys) > 0:
            for k in subkeys:
                if not isinstance(k, AdfKey):
                    raise ValueError('Not all subkeys are ``AdfKey`` objects!')
        self._sized_op = None
        if len(self.options) > 0:
            self._sized_op = isinstance(self.options[0], (list, tuple))

    def _options_string(self):
        """Return the option string."""
        if len(self.options) > 0:
            opt_str = ''
            for op in self.options:
                if self._sized_op:
                    opt_str += f'{op[0]}={op[1]} '
                else:
                    opt_str += f'{op} '
            return opt_str.strip()
        return ''

    def is_block_key(self) -> bool:
        """Return True if this key is a block key."""
        return self.name.upper() in self.block_keys

    @property
    def key(self):
        """
        Return the name of this key. If this is a block key, the name will be
        converted to upper cases.
        """
        if self.is_block_key():
            return self.name.upper()
        return self.name

    def __str__(self):
        """
        Return the string representation of this ``AdfKey``.

        Notes:
            If this key is 'Atoms' and the coordinates are in Cartesian form,
            a different string format will be used.
        """
        adf_str = f'{self.key}'
        if len(self.options) > 0:
            adf_str += f' {self._options_string()}'
        adf_str += '\n'
        if len(self.subkeys) > 0:
            if self.key.lower() == 'atoms':
                for subkey in self.subkeys:
                    adf_str += f'{subkey.name:2s}  {subkey.options[0]: 14.8f}    {subkey.options[1]: 14.8f}    {subkey.options[2]: 14.8f}\n'
            else:
                for subkey in self.subkeys:
                    adf_str += str(subkey)
            if self.is_block_key():
                adf_str += 'END\n'
            else:
                adf_str += 'subend\n'
        elif self.key.upper() in self._full_blocks:
            adf_str += 'END\n'
        return adf_str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdfKey):
            return False
        return str(self) == str(other)

    def has_subkey(self, subkey: str | AdfKey) -> bool:
        """
        Return True if this AdfKey contains the given subkey.

        Args:
            subkey (str or AdfKey): A key name or an AdfKey object.

        Returns:
            bool: Whether this key contains the given key.
        """
        if isinstance(subkey, str):
            key = subkey
        elif isinstance(subkey, AdfKey):
            key = subkey.key
        else:
            raise ValueError('The subkey should be an AdfKey or a string!')
        if len(self.subkeys) > 0 and key in (k.key for k in self.subkeys):
            return True
        return False

    def add_subkey(self, subkey):
        """
        Add a new subkey to this key.

        Args:
            subkey (AdfKey): A new subkey.

        Notes:
            Duplicate check will not be performed if this is an 'Atoms' block.
        """
        if self.key.lower() == 'atoms' or not self.has_subkey(subkey):
            self.subkeys.append(subkey)

    def remove_subkey(self, subkey):
        """
        Remove the given subkey, if existed, from this AdfKey.

        Args:
            subkey (str or AdfKey): The subkey to remove.
        """
        if len(self.subkeys) > 0:
            key = subkey if isinstance(subkey, str) else subkey.key
            for idx, subkey in enumerate(self.subkeys):
                if subkey.key == key:
                    self.subkeys.pop(idx)
                    break

    def add_option(self, option):
        """
        Add a new option to this key.

        Args:
            option : Sized or str or int or float
                A new option to add. This must have the same format
                with existing options.

        Raises:
            TypeError: If the format of the given ``option`` is different.
        """
        if len(self.options) == 0:
            self.options.append(option)
        else:
            sized_op = isinstance(option, (list, tuple))
            if self._sized_op != sized_op:
                raise TypeError('Option type is mismatched!')
            self.options.append(option)

    def remove_option(self, option: str | int) -> None:
        """
        Remove an option.

        Args:
            option (str | int):  The name or index of the option to remove.

        Raises:
            TypeError: If the option has a wrong type.
        """
        if len(self.options) > 0:
            if self._sized_op:
                if not isinstance(option, str):
                    raise TypeError('``option`` should be a name string!')
                for idx, val in enumerate(self.options):
                    if val[0] == option:
                        self.options.pop(idx)
                        break
            else:
                if not isinstance(option, int):
                    raise TypeError('``option`` should be an integer index!')
                self.options.pop(option)

    def has_option(self, option: str) -> bool:
        """
        Return True if the option is included in this key.

        Args:
            option (str): The option.

        Returns:
            bool: Whether the option can be found.
        """
        if len(self.options) == 0:
            return False
        return any((self._sized_op and op[0] == option or op == option for op in self.options))

    def as_dict(self):
        """A JSON-serializable dict representation of self."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'name': self.name, 'options': self.options}
        if len(self.subkeys) > 0:
            subkeys = []
            for subkey in self.subkeys:
                subkeys.append(subkey.as_dict())
            dct['subkeys'] = subkeys
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Construct a MSONable AdfKey object from the JSON dict.

        Args:
            dct (dict): A dict of saved attributes.

        Returns:
            AdfKey: An AdfKey object recovered from the JSON dict.
        """
        key = dct.get('name')
        options = dct.get('options')
        subkey_list = dct.get('subkeys', [])
        subkeys = [AdfKey.from_dict(k) for k in subkey_list] or None
        return cls(key, options, subkeys)

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        Construct an AdfKey object from the string.

        Args:
            string: str

        Returns:
            An AdfKey object recovered from the string.

        Raises:
            ValueError: Currently nested subkeys are not supported.
                If ``subend`` was found a ValueError would be raised.

        Notes:
            Only the first block key will be returned.
        """

        def is_float(s) -> bool:
            return '.' in s or 'E' in s or 'e' in s
        if string.find('\n') == -1:
            el = string.split()
            if len(el) > 1:
                options = [s.split('=') for s in el[1:]] if string.find('=') != -1 else el[1:]
                for idx, op in enumerate(options):
                    if isinstance(op, list) and is_numeric(op[1]):
                        op[1] = float(op[1]) if is_float(op[1]) else int(op[1])
                    elif is_numeric(op):
                        options[idx] = float(op) if is_float(op) else int(op)
            else:
                options = None
            return cls(el[0], options)
        if string.find('subend') != -1:
            raise ValueError('Nested subkeys are not supported!')

        def iter_lines(s: str) -> Generator[str, None, None]:
            """A generator form of s.split('\\n') for reducing memory overhead.

            Args:
                s (str): A multi-line string.

            Yields:
                str: line
            """
            prev_nl = -1
            while True:
                next_nl = s.find('\n', prev_nl + 1)
                if next_nl < 0:
                    yield s[prev_nl + 1:]
                    break
                yield s[prev_nl + 1:next_nl]
                prev_nl = next_nl
        key = None
        for line in iter_lines(string):
            if line == '':
                continue
            el = line.strip().split()
            if len(el) == 0:
                continue
            if el[0].upper() in cls.block_keys:
                if key is None:
                    key = cls.from_str(line)
                else:
                    return key
            elif el[0].upper() == 'END':
                return key
            elif key is not None:
                key.add_subkey(cls.from_str(line))
        raise KeyError("Incomplete key: 'END' is missing!")