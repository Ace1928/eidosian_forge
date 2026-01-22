from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
def _humanText(self):
    return self.stop and self.encoded[1:-1] or self.encoded