import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def StandardFonts(canvas, Write):
    canvas.defaultLineColor = black
    curs = [10, 70]
    for size in (12, 18):
        for fontname in ('times', 'courier', 'helvetica', 'symbol', 'monospaced', 'serif', 'sansserif'):
            curs[0] = 10
            curs[1] = curs[1] + size * 1.5
            Write(canvas, '%s %d ' % (fontname, size), Font(face=fontname, size=size), curs)
            Write(canvas, 'bold ', Font(face=fontname, size=size, bold=1), curs)
            Write(canvas, 'italic ', Font(face=fontname, size=size, italic=1), curs)
            Write(canvas, 'underline', Font(face=fontname, size=size, underline=1), curs)