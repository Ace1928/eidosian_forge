from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def FormResources():
    return PDFPattern(FormResourcesDictionaryPattern, Encoding=PDFPattern(EncodingPattern, PDFDocEncoding=PDFPattern(PDFDocEncodingPattern)), Font=FormFontsDictionary())