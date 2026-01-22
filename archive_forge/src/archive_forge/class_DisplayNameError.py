from Xlib import X
from Xlib.protocol import rq
class DisplayNameError(DisplayError):

    def __str__(self):
        return 'Bad display name "%s"' % self.display