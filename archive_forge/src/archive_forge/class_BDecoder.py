from typing import Dict, Type, Callable, List
class BDecoder(object):

    def __init__(self, yield_tuples=False):
        """Constructor.

        :param yield_tuples: if true, decode "l" elements as tuples rather than
            lists.
        """
        self.yield_tuples = yield_tuples
        decode_func = {}
        decode_func[b'l'] = self.decode_list
        decode_func[b'd'] = self.decode_dict
        decode_func[b'i'] = self.decode_int
        decode_func[b'0'] = self.decode_string
        decode_func[b'1'] = self.decode_string
        decode_func[b'2'] = self.decode_string
        decode_func[b'3'] = self.decode_string
        decode_func[b'4'] = self.decode_string
        decode_func[b'5'] = self.decode_string
        decode_func[b'6'] = self.decode_string
        decode_func[b'7'] = self.decode_string
        decode_func[b'8'] = self.decode_string
        decode_func[b'9'] = self.decode_string
        self.decode_func = decode_func

    def decode_int(self, x, f):
        f += 1
        newf = x.index(b'e', f)
        n = int(x[f:newf])
        if x[f:f + 2] == b'-0':
            raise ValueError
        elif x[f:f + 1] == b'0' and newf != f + 1:
            raise ValueError
        return (n, newf + 1)

    def decode_string(self, x, f):
        colon = x.index(b':', f)
        n = int(x[f:colon])
        if x[f:f + 1] == b'0' and colon != f + 1:
            raise ValueError
        colon += 1
        return (x[colon:colon + n], colon + n)

    def decode_list(self, x, f):
        r, f = ([], f + 1)
        while x[f:f + 1] != b'e':
            v, f = self.decode_func[x[f:f + 1]](x, f)
            r.append(v)
        if self.yield_tuples:
            r = tuple(r)
        return (r, f + 1)

    def decode_dict(self, x, f):
        r, f = ({}, f + 1)
        lastkey = None
        while x[f:f + 1] != b'e':
            k, f = self.decode_string(x, f)
            if lastkey is not None and lastkey >= k:
                raise ValueError
            lastkey = k
            r[k], f = self.decode_func[x[f:f + 1]](x, f)
        return (r, f + 1)

    def bdecode(self, x):
        if not isinstance(x, bytes):
            raise TypeError
        try:
            r, l = self.decode_func[x[:1]](x, 0)
        except (IndexError, KeyError, OverflowError) as e:
            raise ValueError(str(e))
        if l != len(x):
            raise ValueError
        return r