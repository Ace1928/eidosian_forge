from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def FormFont(BaseFont, Name):
    from reportlab.pdfbase.pdfdoc import PDFName
    return PDFPattern(FormFontPattern, BaseFont=PDFName(BaseFont), Name=PDFName(Name), Encoding=PDFPattern(PDFDocEncodingPattern))