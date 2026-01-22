import io
import struct
import typing
def _create_subkeys(self, key: bytes) -> typing.List[typing.List[int]]:
    key_bits = self._get_bits(key)
    pc1_bits = [key_bits[x] for x in self._pc1]
    c = [pc1_bits[0:28]]
    d = [pc1_bits[28:56]]
    for i, shift_index in enumerate(self._shift_indexes):
        c.append(self._shift_bits(c[i], shift_index))
        d.append(self._shift_bits(d[i], shift_index))
    subkeys: typing.List[typing.List[int]] = []
    for i in range(1, 17):
        cd = c[i] + d[i]
        subkey_bits = [cd[x] for x in self._pc2]
        subkeys.append(subkey_bits)
    return subkeys