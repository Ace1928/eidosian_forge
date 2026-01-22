import glob
import os
from io import StringIO
def _AsciiBase85Encode(input):
    """This is a compact encoding used for binary data within
      a PDF file.  Four bytes of binary data become five bytes of
      ASCII.  This is the default method used for encoding images."""
    outstream = StringIO()
    whole_word_count, remainder_size = divmod(len(input), 4)
    cut = 4 * whole_word_count
    body, lastbit = (input[0:cut], input[cut:])
    for i in range(whole_word_count):
        offset = i * 4
        b1 = ord(body[offset])
        b2 = ord(body[offset + 1])
        b3 = ord(body[offset + 2])
        b4 = ord(body[offset + 3])
        num = 16777216 * b1 + 65536 * b2 + 256 * b3 + b4
        if num == 0:
            outstream.write('z')
        else:
            temp, c5 = divmod(num, 85)
            temp, c4 = divmod(temp, 85)
            temp, c3 = divmod(temp, 85)
            c1, c2 = divmod(temp, 85)
            assert 85 ** 4 * c1 + 85 ** 3 * c2 + 85 ** 2 * c3 + 85 * c4 + c5 == num, 'dodgy code!'
            outstream.write(chr(c1 + 33))
            outstream.write(chr(c2 + 33))
            outstream.write(chr(c3 + 33))
            outstream.write(chr(c4 + 33))
            outstream.write(chr(c5 + 33))
    if remainder_size > 0:
        while len(lastbit) < 4:
            lastbit = lastbit + '\x00'
        b1 = ord(lastbit[0])
        b2 = ord(lastbit[1])
        b3 = ord(lastbit[2])
        b4 = ord(lastbit[3])
        num = 16777216 * b1 + 65536 * b2 + 256 * b3 + b4
        temp, c5 = divmod(num, 85)
        temp, c4 = divmod(temp, 85)
        temp, c3 = divmod(temp, 85)
        c1, c2 = divmod(temp, 85)
        lastword = chr(c1 + 33) + chr(c2 + 33) + chr(c3 + 33) + chr(c4 + 33) + chr(c5 + 33)
        outstream.write(lastword[0:remainder_size + 1])
    outstream.write('~>')
    outstream.reset()
    return outstream.read()