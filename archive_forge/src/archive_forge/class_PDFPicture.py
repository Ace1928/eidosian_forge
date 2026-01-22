from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class PDFPicture:

    def __init__(self, canvas, width):
        ulx, uly, lrx, lry = canvas.bbox(Tk_.ALL)
        scale = float(width) / (lrx - ulx)
        pyx.unit.set(uscale=scale, wscale=scale, defaultunit='pt')
        self.transform = lambda xy: (xy[0] - ulx, -xy[1] + lry)
        self.canvas = pyx.canvas.canvas()

    def save(self, file_name):
        page = pyx.document.page(self.canvas, bboxenlarge=3.5 * pyx.unit.t_pt)
        doc = pyx.document.document([page])
        doc.writePDFfile(file_name)