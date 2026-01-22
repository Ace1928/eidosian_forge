import struct
class _PythonARC4(object):
    state = None
    i = 0
    j = 0

    def __init__(self, key):
        key_bytes = []
        for i in range(len(key)):
            key_byte = struct.unpack('B', key[i:i + 1])[0]
            key_bytes.append(key_byte)
        self.state = [n for n in range(256)]
        j = 0
        for i in range(256):
            j = (j + self.state[i] + key_bytes[i % len(key_bytes)]) % 256
            self.state[i], self.state[j] = (self.state[j], self.state[i])

    def update(self, value):
        chars = []
        random_gen = self._random_generator()
        for i in range(len(value)):
            byte = struct.unpack('B', value[i:i + 1])[0]
            updated_byte = byte ^ next(random_gen)
            chars.append(updated_byte)
        return bytes(bytearray(chars))

    def _random_generator(self):
        while True:
            self.i = (self.i + 1) % 256
            self.j = (self.j + self.state[self.i]) % 256
            self.state[self.i], self.state[self.j] = (self.state[self.j], self.state[self.i])
            yield self.state[(self.state[self.i] + self.state[self.j]) % 256]