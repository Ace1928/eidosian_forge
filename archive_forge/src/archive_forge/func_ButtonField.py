from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def ButtonField(title, value, xmin, ymin, page, width=16.7704, height=14.907):
    if value not in ('Yes', 'Off'):
        raise ValueError("button value must be 'Yes' or 'Off': " + repr(value))
    fontSize = 11.3086 / 14.907 * height
    dx = 3.6017 / 16.7704 * width
    dy = 3.3881 / 14.907 * height
    return PDFPattern(ButtonFieldPattern, Name=PDFString(title), xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + width, Hide=PDFPattern(['<< /S  /Hide >>']), APDOff=ButtonStream('0.749 g 0 0 %(width)s %(height)s re f\r\n' % vars(), width=width, height=height), APDYes=ButtonStream('0.749 g 0 0 %(width)s %(height)s re f q 1 1 %(width)s %(height)s re W n BT /ZaDb %(fontSize)s Tf 0 g 1 0 0 1 %(dx)s %(dy)s Tm (4) Tj ET\r\n' % vars(), width=width, height=height), APNYes=ButtonStream('q 1 1 %(width)s %(height)s re W n BT /ZaDb %(fontSize)s Tf 0 g   1 0 0 1 %(dx)s %(dy)s Tm (4) Tj ET Q\r\n' % vars(), width=width, height=height), Value=PDFName(value), Page=page)