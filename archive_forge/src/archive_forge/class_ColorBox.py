import colorsys
import io
from time import process_time
from pip._vendor.rich import box
from pip._vendor.rich.color import Color
from pip._vendor.rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from pip._vendor.rich.markdown import Markdown
from pip._vendor.rich.measure import Measurement
from pip._vendor.rich.pretty import Pretty
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._vendor.rich.syntax import Syntax
from pip._vendor.rich.table import Table
from pip._vendor.rich.text import Text
class ColorBox:

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for y in range(0, 5):
            for x in range(options.max_width):
                h = x / options.max_width
                l = 0.1 + y / 5 * 0.7
                r1, g1, b1 = colorsys.hls_to_rgb(h, l, 1.0)
                r2, g2, b2 = colorsys.hls_to_rgb(h, l + 0.7 / 10, 1.0)
                bgcolor = Color.from_rgb(r1 * 255, g1 * 255, b1 * 255)
                color = Color.from_rgb(r2 * 255, g2 * 255, b2 * 255)
                yield Segment('▄', Style(color=color, bgcolor=bgcolor))
            yield Segment.line()

    def __rich_measure__(self, console: 'Console', options: ConsoleOptions) -> Measurement:
        return Measurement(1, options.max_width)