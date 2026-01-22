from gitdb.util import bin_to_hex
from gitdb.fun import (
class ODeltaPackInfo(OPackInfo):
    """Adds delta specific information,
    Either the 20 byte sha which points to some object in the database,
    or the negative offset from the pack_offset, so that pack_offset - delta_info yields
    the pack offset of the base object"""
    __slots__ = tuple()

    def __new__(cls, packoffset, type, size, delta_info):
        return tuple.__new__(cls, (packoffset, type, size, delta_info))

    @property
    def delta_info(self):
        return self[3]