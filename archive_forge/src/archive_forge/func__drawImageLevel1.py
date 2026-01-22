import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
def _drawImageLevel1(self, image, x1, y1, width=None, height=None):
    component_depth = 8
    myimage = image.convert('RGB')
    imgwidth, imgheight = myimage.size
    if not width:
        width = imgwidth
    if not height:
        height = imgheight
    self.code.extend(['gsave', '%s %s translate' % (x1, y1), '%s %s scale' % (width, height), '/scanline %d 3 mul string def' % imgwidth])
    self.code.extend(['%s %s %s' % (imgwidth, imgheight, component_depth), '[%s %s %s %s %s %s]' % (imgwidth, 0, 0, -imgheight, 0, imgheight), '{ currentfile scanline readhexstring pop } false 3', 'colorimage '])
    rawimage = (myimage.tobytes if hasattr(myimage, 'tobytes') else myimage.tostring)()
    hex_encoded = self._AsciiHexEncode(rawimage)
    outstream = StringIO(hex_encoded)
    dataline = outstream.read(78)
    while dataline != '':
        self.code_append(dataline)
        dataline = outstream.read(78)
    self.code_append('% end of image data')
    self.code_append('grestore')