from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_unicodeTolerance(self) -> None:
    import struct
    s = '<foo><bar><baz /></bar></foo>'
    j = '<?xml version="1.0" encoding="UCS-2" ?>\r\n<JAPANESE>\r\n<TITLE>専門家リスト </TITLE></JAPANESE>'
    j2 = b'\xff\xfe<\x00?\x00x\x00m\x00l\x00 \x00v\x00e\x00r\x00s\x00i\x00o\x00n\x00=\x00"\x001\x00.\x000\x00"\x00 \x00e\x00n\x00c\x00o\x00d\x00i\x00n\x00g\x00=\x00"\x00U\x00C\x00S\x00-\x002\x00"\x00 \x00?\x00>\x00\r\x00\n\x00<\x00J\x00A\x00P\x00A\x00N\x00E\x00S\x00E\x00>\x00\r\x00\n\x00<\x00T\x00I\x00T\x00L\x00E\x00>\x00\x02\\\x80\x95\xb6[\xea0\xb90\xc80 \x00<\x00/\x00T\x00I\x00T\x00L\x00E\x00>\x00<\x00/\x00J\x00A\x00P\x00A\x00N\x00E\x00S\x00E\x00>\x00'

    def reverseBytes(s: bytes) -> bytes:
        fmt = str(len(s) // 2) + 'H'
        return struct.pack('<' + fmt, *struct.unpack('>' + fmt, s))
    urd = microdom.parseString(reverseBytes(s.encode('UTF-16')))
    ud = microdom.parseString(s.encode('UTF-16'))
    sd = microdom.parseString(s)
    self.assertTrue(ud.isEqualToDocument(sd))
    self.assertTrue(ud.isEqualToDocument(urd))
    ud = microdom.parseString(j)
    urd = microdom.parseString(reverseBytes(j2))
    sd = microdom.parseString(j2)
    self.assertTrue(ud.isEqualToDocument(sd))
    self.assertTrue(ud.isEqualToDocument(urd))
    j3 = microdom.parseString('<foo/>')
    hdr = '<?xml version="1.0"?>'
    div = microdom.lmx().text('√', raw=1).node
    de = j3.documentElement
    de.appendChild(div)
    de.appendChild(j3.createComment('√'))
    self.assertEqual(j3.toxml(), hdr + '<foo><div>√</div><!--√--></foo>')