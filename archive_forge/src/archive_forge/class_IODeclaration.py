import enum
from typing import Optional, List, Union, Iterable, Tuple
class IODeclaration(ClassicalDeclaration):
    """A declaration of an IO variable."""

    def __init__(self, modifier: IOModifier, type_: ClassicalType, identifier: Identifier):
        super().__init__(type_, identifier)
        self.modifier = modifier