from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class StructRef(Type):
    """A mutable struct.
    """

    def __init__(self, fields):
        """
        Parameters
        ----------
        fields : Sequence
            A sequence of field descriptions, which is a 2-tuple-like object
            containing `(name, type)`, where `name` is a `str` for the field
            name, and `type` is a numba type for the field type.
        """

        def check_field_pair(fieldpair):
            name, typ = fieldpair
            if not isinstance(name, str):
                msg = 'expecting a str for field name'
                raise ValueError(msg)
            if not isinstance(typ, Type):
                msg = 'expecting a Numba Type for field type'
                raise ValueError(msg)
            return (name, typ)
        fields = tuple(map(check_field_pair, fields))
        self._fields = tuple(map(check_field_pair, self.preprocess_fields(fields)))
        self._typename = self.__class__.__qualname__
        name = f'numba.{self._typename}{self._fields}'
        super().__init__(name=name)

    def preprocess_fields(self, fields):
        """Subclasses can override this to do additional clean up on fields.

        The default is an identity function.

        Parameters:
        -----------
        fields : Sequence[Tuple[str, Type]]
        """
        return fields

    @property
    def field_dict(self):
        """Return an immutable mapping for the field names and their
        corresponding types.
        """
        return MappingProxyType(dict(self._fields))

    def get_data_type(self):
        """Get the payload type for the actual underlying structure referred
        to by this struct reference.

        See also: `ClassInstanceType.get_data_type`
        """
        return StructRefPayload(typename=self.__class__.__name__, fields=self._fields)