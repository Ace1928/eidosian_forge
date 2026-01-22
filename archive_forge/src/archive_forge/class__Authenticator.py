import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
class _Authenticator:
    """Private class to provide en/decoding
            for base64-based authentication conversation.
    """

    def __init__(self, mechinst):
        self.mech = mechinst

    def process(self, data):
        ret = self.mech(self.decode(data))
        if ret is None:
            return b'*'
        return self.encode(ret)

    def encode(self, inp):
        oup = b''
        if isinstance(inp, str):
            inp = inp.encode('utf-8')
        while inp:
            if len(inp) > 48:
                t = inp[:48]
                inp = inp[48:]
            else:
                t = inp
                inp = b''
            e = binascii.b2a_base64(t)
            if e:
                oup = oup + e[:-1]
        return oup

    def decode(self, inp):
        if not inp:
            return b''
        return binascii.a2b_base64(inp)