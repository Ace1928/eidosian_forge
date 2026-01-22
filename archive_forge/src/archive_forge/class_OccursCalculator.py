from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
class OccursCalculator:
    """
    A helper class for adding and multiplying min/max occurrences of XSD particles.
    """
    min_occurs: int
    max_occurs: Optional[int]

    def __init__(self) -> None:
        self.min_occurs = self.max_occurs = 0

    def __repr__(self) -> str:
        return '%s(%r, %r)' % (self.__class__.__name__, self.min_occurs, self.max_occurs)

    def __add__(self, other: ParticleMixin) -> 'OccursCalculator':
        self.min_occurs += other.min_occurs
        if self.max_occurs is not None:
            if other.max_occurs is None:
                self.max_occurs = None
            else:
                self.max_occurs += other.max_occurs
        return self

    def __mul__(self, other: ParticleMixin) -> 'OccursCalculator':
        self.min_occurs *= other.min_occurs
        if self.max_occurs is None:
            if other.max_occurs == 0:
                self.max_occurs = 0
        elif other.max_occurs is None:
            if self.max_occurs != 0:
                self.max_occurs = None
        else:
            self.max_occurs *= other.max_occurs
        return self

    def reset(self) -> None:
        self.min_occurs = self.max_occurs = 0