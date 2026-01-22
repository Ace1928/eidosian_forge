from . import DefaultTable
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import struct

        Calculate offsets to VDMX_Group records.
        For each ratRange return a list of offset values from the beginning of
        the VDMX table to a VDMX_Group.
        