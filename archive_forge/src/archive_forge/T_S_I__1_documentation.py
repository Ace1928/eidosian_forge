from . import DefaultTable
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import strjoin, tobytes, tostr
 TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI1 contains the text of the glyph programs in the form of low-level assembly
code, as well as the 'extra' programs 'fpgm', 'ppgm' (i.e. 'prep'), and 'cvt'.
