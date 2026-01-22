from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
class TableStyle:

    def __init__(self, cmds=None, parent=None, **kw):
        if parent:
            pcmds = parent.getCommands()[:]
            self._opts = parent._opts
            for a in ('spaceBefore', 'spaceAfter'):
                if hasattr(parent, a):
                    setattr(self, a, getattr(parent, a))
        else:
            pcmds = []
        self._cmds = pcmds + list(cmds or [])
        self._opts = {}
        self._opts.update(kw)

    def add(self, *cmd):
        self._cmds.append(cmd)

    def __repr__(self):
        return 'TableStyle(\n%s\n) # end TableStyle' % '  \n'.join(map(repr, self._cmds))

    def getCommands(self):
        return self._cmds