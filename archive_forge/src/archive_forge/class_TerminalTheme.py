from typing import List, Optional, Tuple
from .color_triplet import ColorTriplet
from .palette import Palette
class TerminalTheme:
    """A color theme used when exporting console content.

    Args:
        background (Tuple[int, int, int]): The background color.
        foreground (Tuple[int, int, int]): The foreground (text) color.
        normal (List[Tuple[int, int, int]]): A list of 8 normal intensity colors.
        bright (List[Tuple[int, int, int]], optional): A list of 8 bright colors, or None
            to repeat normal intensity. Defaults to None.
    """

    def __init__(self, background: _ColorTuple, foreground: _ColorTuple, normal: List[_ColorTuple], bright: Optional[List[_ColorTuple]]=None) -> None:
        self.background_color = ColorTriplet(*background)
        self.foreground_color = ColorTriplet(*foreground)
        self.ansi_colors = Palette(normal + (bright or normal))