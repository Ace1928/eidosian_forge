from fontTools.misc.roundTools import otRound, roundFunc
from fontTools.misc.psCharStrings import T2CharString
from fontTools.pens.basePen import BasePen
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
Pen to draw Type 2 CharStrings.

    The 'roundTolerance' argument controls the rounding of point coordinates.
    It is defined as the maximum absolute difference between the original
    float and the rounded integer value.
    The default tolerance of 0.5 means that all floats are rounded to integer;
    a value of 0 disables rounding; values in between will only round floats
    which are close to their integral part within the tolerated range.
    