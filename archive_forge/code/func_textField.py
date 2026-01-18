from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def textField(self, canvas, title, xmin, ymin, xmax, ymax, value='', maxlen=1000000, multiline=0):
    doc = canvas._doc
    page = doc.thisPageRef()
    R, G, B = obj_R_G_B(canvas._fillColorObj)
    font = canvas._fontname
    fontsize = canvas._fontsize
    field = TextField(title, value, xmin, ymin, xmax, ymax, page, maxlen, font, fontsize, R, G, B, multiline)
    self.fields.append(field)
    canvas._addAnnotation(field)