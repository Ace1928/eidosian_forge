from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class _Strip:
    __slots__ = ('x', 'y', 'max_height', 'y2')

    def __init__(self, y: int, max_height: int) -> None:
        self.x = 0
        self.y = y
        self.max_height = max_height
        self.y2 = y

    def add(self, width: int, height: int) -> Tuple[int, int]:
        assert width > 0 and height > 0
        assert height <= self.max_height
        x, y = (self.x, self.y)
        self.x += width
        self.y2 = max(self.y + height, self.y2)
        return (x, y)

    def compact(self) -> None:
        self.max_height = self.y2 - self.y