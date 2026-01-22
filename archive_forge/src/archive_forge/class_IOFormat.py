import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
class IOFormat:

    def __init__(self, name: str, desc: str, code: str, module_name: str, encoding: str=None) -> None:
        self.name = name
        self.description = desc
        assert len(code) == 2
        assert code[0] in list('+1')
        assert code[1] in list('BFS')
        self.code = code
        self.module_name = module_name
        self.encoding = encoding
        self.extensions: List[str] = []
        self.globs: List[str] = []
        self.magic: List[str] = []
        self.magic_regex: Optional[bytes] = None

    def open(self, fname, mode: str='r') -> IO:
        if mode not in list('rwa'):
            raise ValueError("Only modes allowed are 'r', 'w', and 'a'")
        if mode == 'r' and (not self.can_read):
            raise NotImplementedError('No reader implemented for {} format'.format(self.name))
        if mode == 'w' and (not self.can_write):
            raise NotImplementedError('No writer implemented for {} format'.format(self.name))
        if mode == 'a' and (not self.can_append):
            raise NotImplementedError('Appending not supported by {} format'.format(self.name))
        if self.isbinary:
            mode += 'b'
        path = Path(fname)
        return path.open(mode, encoding=self.encoding)

    def _buf_as_filelike(self, data: Union[str, bytes]) -> IO:
        encoding = self.encoding
        if encoding is None:
            encoding = 'utf-8'
        if self.isbinary:
            if isinstance(data, str):
                data = data.encode(encoding)
        elif isinstance(data, bytes):
            data = data.decode(encoding)
        return self._ioclass(data)

    @property
    def _ioclass(self):
        if self.isbinary:
            return io.BytesIO
        else:
            return io.StringIO

    def parse_images(self, data: Union[str, bytes], **kwargs) -> Sequence[Atoms]:
        with self._buf_as_filelike(data) as fd:
            outputs = self.read(fd, **kwargs)
            if self.single:
                assert isinstance(outputs, Atoms)
                return [outputs]
            else:
                return list(self.read(fd, **kwargs))

    def parse_atoms(self, data: Union[str, bytes], **kwargs) -> Atoms:
        images = self.parse_images(data, **kwargs)
        return images[-1]

    @property
    def can_read(self) -> bool:
        return self._readfunc() is not None

    @property
    def can_write(self) -> bool:
        return self._writefunc() is not None

    @property
    def can_append(self) -> bool:
        writefunc = self._writefunc()
        return self.can_write and 'append' in writefunc.__code__.co_varnames

    def __repr__(self) -> str:
        tokens = ['{}={}'.format(name, repr(value)) for name, value in vars(self).items()]
        return 'IOFormat({})'.format(', '.join(tokens))

    def __getitem__(self, i):
        return (self.description, self.code)[i]

    @property
    def single(self) -> bool:
        """Whether this format is for a single Atoms object."""
        return self.code[0] == '1'

    @property
    def _formatname(self) -> str:
        return self.name.replace('-', '_')

    def _readfunc(self):
        return getattr(self.module, 'read_' + self._formatname, None)

    def _writefunc(self):
        return getattr(self.module, 'write_' + self._formatname, None)

    @property
    def read(self):
        if not self.can_read:
            self._warn_none('read')
            return None
        return self._read_wrapper

    def _read_wrapper(self, *args, **kwargs):
        function = self._readfunc()
        if function is None:
            self._warn_none('read')
            return None
        if not inspect.isgeneratorfunction(function):
            function = functools.partial(wrap_read_function, function)
        return function(*args, **kwargs)

    def _warn_none(self, action):
        msg = 'Accessing the IOFormat.{action} property on a format without {action} support will change behaviour in the future and return a callable instead of None.  Use IOFormat.can_{action} to check whether {action} is supported.'
        warnings.warn(msg.format(action=action), FutureWarning)

    @property
    def write(self):
        if not self.can_write:
            self._warn_none('write')
            return None
        return self._write_wrapper

    def _write_wrapper(self, *args, **kwargs):
        function = self._writefunc()
        if function is None:
            raise ValueError(f'Cannot write to {self.name}-format')
        return function(*args, **kwargs)

    @property
    def modes(self) -> str:
        modes = ''
        if self.can_read:
            modes += 'r'
        if self.can_write:
            modes += 'w'
        return modes

    def full_description(self) -> str:
        lines = [f'Name:        {self.name}', f'Description: {self.description}', f'Modes:       {self.modes}', f'Encoding:    {self.encoding}', f'Module:      {self.module_name}', f'Code:        {self.code}', f'Extensions:  {self.extensions}', f'Globs:       {self.globs}', f'Magic:       {self.magic}']
        return '\n'.join(lines)

    @property
    def acceptsfd(self) -> bool:
        return self.code[1] != 'S'

    @property
    def isbinary(self) -> bool:
        return self.code[1] == 'B'

    @property
    def module(self):
        if not self.module_name.startswith('ase.io.'):
            raise ValueError('Will only import modules from ase.io, not {}'.format(self.module_name))
        try:
            return import_module(self.module_name)
        except ImportError as err:
            raise UnknownFileTypeError(f'File format not recognized: {self.name}.  Error: {err}')

    def match_name(self, basename: str) -> bool:
        from fnmatch import fnmatch
        return any((fnmatch(basename, pattern) for pattern in self.globs))

    def match_magic(self, data: bytes) -> bool:
        if self.magic_regex:
            assert not self.magic, 'Define only one of magic and magic_regex'
            match = re.match(self.magic_regex, data, re.M | re.S)
            return match is not None
        from fnmatch import fnmatchcase
        return any((fnmatchcase(data, magic + b'*') for magic in self.magic))