import os
from typing import Optional, Union
from warnings import warn
from zope.interface import implementer
import attr
from typing_extensions import Literal
from twisted.internet.interfaces import IAddress
from twisted.python.filepath import _asFilesystemBytes, _coerceToFilesystemEncoding
from twisted.python.runtime import platform
@attr.s(hash=False, repr=False, eq=False, auto_attribs=True)
@implementer(IAddress)
class UNIXAddress:
    """
    Object representing a UNIX socket endpoint.

    @ivar name: The filename associated with this socket.
    @type name: C{bytes}
    """
    name: Optional[bytes] = attr.ib(converter=attr.converters.optional(_asFilesystemBytes))
    if getattr(os.path, 'samefile', None) is not None:

        def __eq__(self, other: object) -> bool:
            """
            Overriding C{attrs} to ensure the os level samefile
            check is done if the name attributes do not match.
            """
            if not isinstance(other, self.__class__):
                return NotImplemented
            res = self.name == other.name
            if not res and self.name and other.name:
                try:
                    return os.path.samefile(self.name, other.name)
                except OSError:
                    pass
                except (TypeError, ValueError) as e:
                    if not platform.isLinux():
                        raise e
            return res
    else:

        def __eq__(self, other: object) -> bool:
            if isinstance(other, self.__class__):
                return self.name == other.name
            return NotImplemented

    def __repr__(self) -> str:
        name = self.name
        show = _coerceToFilesystemEncoding('', name) if name is not None else None
        return f'UNIXAddress({show!r})'

    def __hash__(self):
        if self.name is None:
            return hash((self.__class__, None))
        try:
            s1 = os.stat(self.name)
            return hash((s1.st_ino, s1.st_dev))
        except OSError:
            return hash(self.name)