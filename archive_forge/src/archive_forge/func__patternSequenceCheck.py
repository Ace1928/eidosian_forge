from reportlab.pdfbase.pdfdoc import format, PDFObject, pdfdocEnc
from reportlab.lib.utils import strTypes
def _patternSequenceCheck(pattern_sequence):
    allowedTypes = strTypes if isinstance(strTypes, tuple) else (strTypes,)
    allowedTypes = allowedTypes + (PDFObject, PDFPatternIf)
    for x in pattern_sequence:
        if not isinstance(x, allowedTypes):
            if len(x) != 1:
                raise ValueError('sequence elts must be strings/bytes/PDFPatternIfs or singletons containing strings: ' + ascii(x))
            if not isinstance(x[0], strTypes):
                raise ValueError('Singletons must contain strings/bytes or PDFObject instances only: ' + ascii(x[0]))