from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Type, Union
from uuid import UUID
def as_uuid(self, uuid_representation: int=UuidRepresentation.STANDARD) -> UUID:
    """Create a Python UUID from this BSON Binary object.

        Decodes this binary object as a native :class:`uuid.UUID` instance
        with the provided ``uuid_representation``.

        Raises :exc:`ValueError` if this :class:`~bson.binary.Binary` instance
        does not contain a UUID.

        :Parameters:
          - `uuid_representation`: A member of
            :class:`~bson.binary.UuidRepresentation`. Default:
            :const:`~bson.binary.UuidRepresentation.STANDARD`.
            See :ref:`handling-uuid-data-example` for details.

        .. versionadded:: 3.11
        """
    if self.subtype not in ALL_UUID_SUBTYPES:
        raise ValueError(f'cannot decode subtype {self.subtype} as a uuid')
    if uuid_representation not in ALL_UUID_REPRESENTATIONS:
        raise ValueError('uuid_representation must be a value from bson.binary.UuidRepresentation')
    if uuid_representation == UuidRepresentation.UNSPECIFIED:
        raise ValueError('uuid_representation cannot be UNSPECIFIED')
    elif uuid_representation == UuidRepresentation.PYTHON_LEGACY:
        if self.subtype == OLD_UUID_SUBTYPE:
            return UUID(bytes=self)
    elif uuid_representation == UuidRepresentation.JAVA_LEGACY:
        if self.subtype == OLD_UUID_SUBTYPE:
            return UUID(bytes=self[0:8][::-1] + self[8:16][::-1])
    elif uuid_representation == UuidRepresentation.CSHARP_LEGACY:
        if self.subtype == OLD_UUID_SUBTYPE:
            return UUID(bytes_le=self)
    elif self.subtype == UUID_SUBTYPE:
        return UUID(bytes=self)
    raise ValueError(f'cannot decode subtype {self.subtype} to {UUID_REPRESENTATION_NAMES[uuid_representation]}')