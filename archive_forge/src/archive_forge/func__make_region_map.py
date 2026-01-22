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
def _make_region_map(self, width: int, height: int) -> RegionMap:
    """Create a dict that maps layout on to Region."""
    stack: List[Tuple[Layout, Region]] = [(self, Region(0, 0, width, height))]
    push = stack.append
    pop = stack.pop
    layout_regions: List[Tuple[Layout, Region]] = []
    append_layout_region = layout_regions.append
    while stack:
        append_layout_region(pop())
        layout, region = layout_regions[-1]
        children = layout.children
        if children:
            for child_and_region in layout.splitter.divide(children, region):
                push(child_and_region)
    region_map = {layout: region for layout, region in sorted(layout_regions, key=itemgetter(1))}
    return region_map