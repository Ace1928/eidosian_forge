import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class Validator:
    """base validator class"""

    def __call__(self, x):
        return self.test(x)

    def __str__(self):
        return getattr(self, '_str', self.__class__.__name__)

    def normalize(self, x):
        return x

    def normalizeTest(self, x):
        try:
            self.normalize(x)
            return True
        except:
            return False