from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
class GraphVScale(Widget):
    _sizing = frozenset([Sizing.BOX])

    def __init__(self, labels, top: float) -> None:
        """
        GraphVScale( [(label1 position, label1 markup),...], top )
        label position -- 0 < position < top for the y position
        label markup -- text markup for this label
        top -- top y position

        This widget is a vertical scale for the BarGraph widget that
        can correspond to the BarGraph's horizontal lines
        """
        super().__init__()
        self.set_scale(labels, top)

    def set_scale(self, labels, top: float) -> None:
        """
        set_scale( [(label1 position, label1 markup),...], top )
        label position -- 0 < position < top for the y position
        label markup -- text markup for this label
        top -- top y position
        """
        labels = sorted(labels[:], reverse=True)
        self.pos = []
        self.txt = []
        for y, markup in labels:
            self.pos.append(y)
            self.txt.append(Text(markup))
        self.top = top

    def selectable(self) -> Literal[False]:
        """
        Return False.
        """
        return False

    def render(self, size: tuple[int, int], focus: bool=False) -> SolidCanvas | CompositeCanvas:
        """
        Render GraphVScale.
        """
        maxcol, maxrow = size
        pl = scale_bar_values(self.pos, self.top, maxrow)
        combinelist = []
        rows = 0
        for p, t in zip(pl, self.txt):
            p -= 1
            if p >= maxrow:
                break
            if p < rows:
                continue
            c = t.render((maxcol,))
            if p > rows:
                run = p - rows
                c = CompositeCanvas(c)
                c.pad_trim_top_bottom(run, 0)
            rows += c.rows()
            combinelist.append((c, None, False))
        if not combinelist:
            return SolidCanvas(' ', size[0], size[1])
        canvas = CanvasCombine(combinelist)
        if maxrow - rows:
            canvas.pad_trim_top_bottom(0, maxrow - rows)
        return canvas