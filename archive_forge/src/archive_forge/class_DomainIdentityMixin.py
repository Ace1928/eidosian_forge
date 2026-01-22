from __future__ import annotations
class DomainIdentityMixin:
    __slots__ = ()
    id: int | None
    name: str | None

    @property
    def id_or_name(self) -> int | str:
        """
        Return the first defined value, and fails if none is defined.
        """
        if self.id is not None:
            return self.id
        if self.name is not None:
            return self.name
        raise ValueError('id or name must be set')