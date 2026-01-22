from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
class AbstractDrawer:
    """Abstract Drawer.

    Attributes:
     - tracklines    Boolean for whether to draw lines delineating tracks
     - pagesize      Tuple describing the size of the page in pixels
     - x0            Float X co-ord for leftmost point of drawable area
     - xlim          Float X co-ord for rightmost point of drawable area
     - y0            Float Y co-ord for lowest point of drawable area
     - ylim          Float Y co-ord for topmost point of drawable area
     - pagewidth     Float pixel width of drawable area
     - pageheight    Float pixel height of drawable area
     - xcenter       Float X co-ord of center of drawable area
     - ycenter       Float Y co-ord of center of drawable area
     - start         Int, base to start drawing from
     - end           Int, base to stop drawing at
     - length        Size of sequence to be drawn
     - cross_track_links List of tuples each with four entries (track A,
       feature A, track B, feature B) to be linked.

    """

    def __init__(self, parent, pagesize='A3', orientation='landscape', x=0.05, y=0.05, xl=None, xr=None, yt=None, yb=None, start=None, end=None, tracklines=0, cross_track_links=None):
        """Create the object.

        Arguments:
         - parent    Diagram object containing the data that the drawer draws
         - pagesize  String describing the ISO size of the image, or a tuple
           of pixels
         - orientation   String describing the required orientation of the
           final drawing ('landscape' or 'portrait')
         - x         Float (0->1) describing the relative size of the X
           margins to the page
         - y         Float (0->1) describing the relative size of the Y
           margins to the page
         - xl        Float (0->1) describing the relative size of the left X
           margin to the page (overrides x)
         - xr        Float (0->1) describing the relative size of the right X
           margin to the page (overrides x)
         - yt        Float (0->1) describing the relative size of the top Y
           margin to the page (overrides y)
         - yb        Float (0->1) describing the relative size of the lower Y
           margin to the page (overrides y)
         - start     Int, the position to begin drawing the diagram at
         - end       Int, the position to stop drawing the diagram at
         - tracklines    Boolean flag to show (or not) lines delineating tracks
           on the diagram
         - cross_track_links List of tuples each with four entries (track A,
           feature A, track B, feature B) to be linked.

        """
        self._parent = parent
        self.set_page_size(pagesize, orientation)
        self.set_margins(x, y, xl, xr, yt, yb)
        self.set_bounds(start, end)
        self.tracklines = tracklines
        if cross_track_links is None:
            cross_track_links = []
        else:
            self.cross_track_links = cross_track_links

    def set_page_size(self, pagesize, orientation):
        """Set page size of the drawing..

        Arguments:
         - pagesize      Size of the output image, a tuple of pixels (width,
           height, or a string in the reportlab.lib.pagesizes
           set of ISO sizes.
         - orientation   String: 'landscape' or 'portrait'

        """
        if isinstance(pagesize, str):
            pagesize = page_sizes(pagesize)
        elif isinstance(pagesize, tuple):
            pass
        else:
            raise ValueError(f'Page size {pagesize} not recognised')
        shortside, longside = (min(pagesize), max(pagesize))
        orientation = orientation.lower()
        if orientation not in ('landscape', 'portrait'):
            raise ValueError(f'Orientation {orientation} not recognised')
        if orientation == 'landscape':
            self.pagesize = (longside, shortside)
        else:
            self.pagesize = (shortside, longside)

    def set_margins(self, x, y, xl, xr, yt, yb):
        """Set page margins.

        Arguments:
         - x         Float(0->1), Absolute X margin as % of page
         - y         Float(0->1), Absolute Y margin as % of page
         - xl        Float(0->1), Left X margin as % of page
         - xr        Float(0->1), Right X margin as % of page
         - yt        Float(0->1), Top Y margin as % of page
         - yb        Float(0->1), Bottom Y margin as % of page

        Set the page margins as proportions of the page 0->1, and also
        set the page limits x0, y0 and xlim, ylim, and page center
        xorigin, yorigin, as well as overall page width and height
        """
        xmargin_l = xl or x
        xmargin_r = xr or x
        ymargin_top = yt or y
        ymargin_btm = yb or y
        self.x0, self.y0 = (self.pagesize[0] * xmargin_l, self.pagesize[1] * ymargin_btm)
        self.xlim, self.ylim = (self.pagesize[0] * (1 - xmargin_r), self.pagesize[1] * (1 - ymargin_top))
        self.pagewidth = self.xlim - self.x0
        self.pageheight = self.ylim - self.y0
        self.xcenter, self.ycenter = (self.x0 + self.pagewidth / 2.0, self.y0 + self.pageheight / 2.0)

    def set_bounds(self, start, end):
        """Set start and end points for the drawing as a whole.

        Arguments:
         - start - The first base (or feature mark) to draw from
         - end - The last base (or feature mark) to draw to

        """
        low, high = self._parent.range()
        if start is not None and end is not None and (start > end):
            start, end = (end, start)
        if start is None or start < 0:
            start = 0
        if end is None or end < 0:
            end = high + 1
        self.start, self.end = (int(start), int(end))
        self.length = self.end - self.start + 1

    def is_in_bounds(self, value):
        """Check if given value is within the region selected for drawing.

        Arguments:
         - value - A base position

        """
        if value >= self.start and value <= self.end:
            return 1
        return 0

    def __len__(self):
        """Return the length of the region to be drawn."""
        return self.length

    def _current_track_start_end(self):
        track = self._parent[self.current_track_level]
        if track.start is None:
            start = self.start
        else:
            start = max(self.start, track.start)
        if track.end is None:
            end = self.end
        else:
            end = min(self.end, track.end)
        return (start, end)