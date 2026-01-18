import ast
import string
Pad byte string 'data' with null bytes until its length is a
    multiple of 'size'.

    >>> len(pad(b'abcd', 4))
    4
    >>> len(pad(b'abcde', 2))
    6
    >>> len(pad(b'abcde', 4))
    8
    >>> pad(b'abcdef', 4) == b'abcdef\x00\x00'
    True
    