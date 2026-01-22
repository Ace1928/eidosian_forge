import string
from twisted.logger import Logger
class ColorText:
    """
    Represents an element of text along with the texts colors and
    additional attributes.
    """
    COLORS = ('b', 'r', 'g', 'y', 'l', 'm', 'c', 'w')
    BOLD_COLORS = tuple((x.upper() for x in COLORS))
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(len(COLORS))
    COLOR_NAMES = ('Black', 'Red', 'Green', 'Yellow', 'Blue', 'Magenta', 'Cyan', 'White')

    def __init__(self, text, fg, bg, display, bold, underline, flash, reverse):
        self.text, self.fg, self.bg = (text, fg, bg)
        self.display = display
        self.bold = bold
        self.underline = underline
        self.flash = flash
        self.reverse = reverse
        if self.reverse:
            self.fg, self.bg = (self.bg, self.fg)