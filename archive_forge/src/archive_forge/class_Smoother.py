from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class Smoother:
    """
    An object that displays a smooth link image on a Tk canvas.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.canvas_items = []
        self.curves = []

    def _build_curves(self):
        self.curves = curves = []
        self.polygons = []
        for polyline, color in self.polylines:
            n = len(curves)
            polygon = []
            for arc in polyline:
                polygon += arc[1:-1]
                if arc[0] == arc[-1]:
                    A = SmoothLoop(self.canvas, arc, color, tension1=self.tension1, tension2=self.tension2)
                    curves.append(A)
                else:
                    A = SmoothArc(self.canvas, arc, color, tension1=self.tension1, tension2=self.tension2)
                    curves.append(A)
            self.polygons.append(polygon)

    def set_polylines(self, polylines, thickness=5, tension1=1.0, tension2=1.0):
        self.clear()
        self.polylines = polylines
        self.vertices = []
        self.tension1 = tension1
        self.tension2 = tension2
        self._build_curves()
        self.draw(thickness=thickness)

    def draw(self, thickness=5):
        for curve in self.curves:
            curve.tk_draw(thickness=thickness)

    def clear(self):
        for curve in self.curves:
            curve.tk_clear()

    def save_as_pdf(self, file_name, colormode='color', width=312.0):
        """
        Save the smooth link diagram as a PDF file.
        Accepts options colormode and width.
        The colormode (currently ignored) must be 'color', 'gray', or 'mono'; default is 'color'.
        The width option sets the width of the figure in points.
        The default width is 312pt = 4.33in = 11cm .
        """
        PDF = PDFPicture(self.canvas, width)
        for curve in self.curves:
            curve.pyx_draw(PDF.canvas, PDF.transform)
        PDF.save(file_name)

    def save_as_eps(self, file_name, colormode='color', width=312.0):
        """
        Save the link diagram as an encapsulated postscript file.
        Accepts options colormode and width.
        The colormode must be 'color', 'gray', or 'mono'; default is 'color'.
        The width option sets the width of the figure in points.
        The default width is 312pt = 4.33in = 11cm .
        """
        save_as_eps(self.canvas, file_name, colormode, width)

    def save_as_svg(self, file_name, colormode='color', width=None):
        """
        The colormode (currently ignored) must be 'color', 'gray', or 'mono'.
        The width option is ignored for svg images.
        """
        save_as_svg(self.canvas, file_name, colormode, width)

    def save_as_tikz(self, file_name, colormode='color', width=282.0):
        colors = [pl[-1] for pl in self.polylines]
        tikz = TikZPicture(self.canvas, colors, width)
        for curve in self.curves:
            curve.tikz_draw(tikz, tikz.transform)
        tikz.save(file_name)