import _markupbase
import re
def unknown_charref(self, ref):
    self.flush()
    print('*** unknown char ref: &#' + ref + ';')