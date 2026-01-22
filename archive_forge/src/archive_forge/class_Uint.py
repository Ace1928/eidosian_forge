from __future__ import annotations
import typing
@typing.final
class Uint(Type):
    """An unsigned integer of fixed bit width."""
    __slots__ = ('width',)

    def __init__(self, width: int):
        if isinstance(width, int) and width <= 0:
            raise ValueError('uint width must be greater than zero')
        super(Type, self).__setattr__('width', width)

    def __repr__(self):
        return f'Uint({self.width})'

    def __hash__(self):
        return hash((self.__class__, self.width))

    def __eq__(self, other):
        return isinstance(other, Uint) and self.width == other.width