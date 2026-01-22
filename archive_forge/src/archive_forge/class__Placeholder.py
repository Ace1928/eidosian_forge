from abc import ABC, abstractmethod
from itertools import islice
from operator import itemgetter
from threading import RLock
from typing import (
from ._ratio import ratio_resolve
from .align import Align
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .highlighter import ReprHighlighter
from .panel import Panel
from .pretty import Pretty
from .region import Region
from .repr import Result, rich_repr
from .segment import Segment
from .style import StyleType
class _Placeholder:
    """An internal renderable used as a Layout placeholder."""
    highlighter = ReprHighlighter()

    def __init__(self, layout: 'Layout', style: StyleType='') -> None:
        self.layout = layout
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        height = options.height or options.size.height
        layout = self.layout
        title = f'{layout.name!r} ({width} x {height})' if layout.name else f'({width} x {height})'
        yield Panel(Align.center(Pretty(layout), vertical='middle'), style=self.style, title=self.highlighter(title), border_style='blue', height=height)