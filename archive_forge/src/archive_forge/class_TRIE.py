import os, sys, time, random
class TRIE:
    __slots__ = tuple('ch,to,values,front'.split(','))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr != None:
            if fr.ch != None:
                ret.append(fr.ch)
            fr = fr.front
        return '<TRIE %s %s>' % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int=0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int=0):
        u: TRIE = self
        ch: int = key[idx]
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = (idx, u, u.values)
            if idx == len(key):
                break
            ch = key[idx]
        return ret