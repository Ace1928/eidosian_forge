from __future__ import unicode_literals
from . import (lookup, LABELS, decode, encode, iter_decode, iter_encode,
def test_x_user_defined():
    encoded = b'2,\x0c\x0b\x1aO\xd9#\xcb\x0f\xc9\xbbt\xcf\xa8\xca'
    decoded = '2,\x0c\x0b\x1aO\uf7d9#\uf7cb\x0f\uf7c9\uf7bbt\uf7cf\uf7a8\uf7ca'
    encoded = b'aa'
    decoded = 'aa'
    assert decode(encoded, 'x-user-defined') == (decoded, lookup('x-user-defined'))
    assert encode(decoded, 'x-user-defined') == encoded