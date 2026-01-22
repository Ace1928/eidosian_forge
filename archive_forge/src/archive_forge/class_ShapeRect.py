class ShapeRect(Shape):
    """Class for the representation of a rectangle."""
    __slots__ = ('width', 'height')

    def __init__(self):
        super(ShapeRect, self).__init__()
        self.width = 0
        self.height = 0