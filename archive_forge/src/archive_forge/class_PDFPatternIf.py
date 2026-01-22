from reportlab.pdfbase.pdfdoc import format, PDFObject, pdfdocEnc
from reportlab.lib.utils import strTypes
class PDFPatternIf:
    """cond will be evaluated as [cond] in PDFpattern eval.
    It should evaluate to a list with value 0/1 etc etc.
    thenPart is a list to be evaluated if the cond evaulates true,
    elsePart is the false sequence.
    """

    def __init__(self, cond, thenPart=[], elsePart=[]):
        if not isinstance(cond, list):
            cond = [cond]
        for x in (cond, thenPart, elsePart):
            _patternSequenceCheck(x)
        self.cond = cond
        self.thenPart = thenPart
        self.elsePart = elsePart