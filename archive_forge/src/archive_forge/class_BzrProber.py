from typing import TYPE_CHECKING
from .. import config, controldir, errors, pyutils, registry
from .. import transport as _mod_transport
from ..branch import format_registry as branch_format_registry
from ..repository import format_registry as repository_format_registry
from ..workingtree import format_registry as workingtree_format_registry
class BzrProber(controldir.Prober):
    """Prober for formats that use a .bzr/ control directory."""
    formats = registry.FormatRegistry['BzrDirFormat'](controldir.network_format_registry)
    'The known .bzr formats.'

    @classmethod
    def priority(klass, transport):
        return 10

    @classmethod
    def probe_transport(klass, transport):
        """Return the .bzrdir style format present in a directory."""
        try:
            format_string = transport.get_bytes('.bzr/branch-format')
        except _mod_transport.NoSuchFile:
            raise errors.NotBranchError(path=transport.base)
        except errors.BadHttpRequest as e:
            if e.reason == 'no such method: .bzr':
                raise errors.NotBranchError(path=transport.base)
            raise
        try:
            first_line = format_string[:format_string.index(b'\n') + 1]
        except ValueError:
            first_line = format_string
        if first_line.startswith(b'<!DOCTYPE') or first_line.startswith(b'<html'):
            raise errors.NotBranchError(path=transport.base, detail='format file looks like HTML')
        try:
            cls = klass.formats.get(first_line)
        except KeyError:
            if first_line.endswith(b'\r\n'):
                raise LineEndingError(file='.bzr/branch-format')
            else:
                raise errors.UnknownFormatError(format=first_line, kind='bzrdir')
        return cls.from_string(format_string)

    @classmethod
    def known_formats(cls):
        result = []
        for name, format in cls.formats.items():
            if callable(format):
                format = format()
            result.append(format)
        return result