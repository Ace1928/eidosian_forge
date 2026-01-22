import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isListOfShapes(Validator):
    """ListOfShapes validator class."""

    def test(self, x):
        from reportlab.graphics.shapes import Shape
        if isSeq(x):
            answer = 1
            for e in x:
                if not isinstance(e, Shape):
                    answer = 0
            return answer
        else:
            return False