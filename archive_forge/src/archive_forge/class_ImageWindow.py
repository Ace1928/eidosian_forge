from __future__ import annotations
from . import Image
class ImageWindow(Window):
    """Create an image window which displays the given image."""

    def __init__(self, image, title='PIL'):
        if not isinstance(image, Dib):
            image = Dib(image)
        self.image = image
        width, height = image.size
        super().__init__(title, width=width, height=height)

    def ui_handle_repair(self, dc, x0, y0, x1, y1):
        self.image.draw(dc, (x0, y0, x1, y1))