import codecs
import zlib # this codec needs the optional zlib module !
def zlib_decode(input, errors='strict'):
    assert errors == 'strict'
    return (zlib.decompress(input), len(input))