from __future__ import annotations
class BaseDomain:
    __slots__ = ()

    @classmethod
    def from_dict(cls, data: dict):
        """
        Build the domain object from the data dict.
        """
        supported_data = {k: v for k, v in data.items() if k in cls.__slots__}
        return cls(**supported_data)

    def __repr__(self) -> str:
        kwargs = [f'{key}={getattr(self, key)!r}' for key in self.__slots__]
        return f'{self.__class__.__qualname__}({', '.join(kwargs)})'