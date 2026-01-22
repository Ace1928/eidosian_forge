from .py3compat import byte2int, int2byte, bytes2str
class HexString(bytes):
    """
    Represents bytes that will be hex-dumped to a string when its string
    representation is requested.
    """

    def __init__(self, data, linesize=16):
        self.linesize = linesize

    def __new__(cls, data, *args, **kwargs):
        return bytes.__new__(cls, data)

    def __str__(self):
        if not self:
            return "''"
        sep = '\n'
        return sep + sep.join(hexdump(self, self.linesize))