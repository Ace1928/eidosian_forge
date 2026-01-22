from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def ButtonStream(content, width=16.7704, height=14.907):
    result = PDFStream(buttonStreamDictionary(width=width, height=height), content)
    result.filters = []
    return result