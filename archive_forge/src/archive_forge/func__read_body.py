from testtools.compat import _b
def _read_body(self):
    """Pass body bytes to the output."""
    while self.body_length and self.buffered_bytes:
        if self.body_length >= len(self.buffered_bytes[0]):
            self.output.write(self.buffered_bytes[0])
            self.body_length -= len(self.buffered_bytes[0])
            del self.buffered_bytes[0]
            if not self.body_length:
                self.state = self._read_length
        else:
            self.output.write(self.buffered_bytes[0][:self.body_length])
            self.buffered_bytes[0] = self.buffered_bytes[0][self.body_length:]
            self.body_length = 0
            self.state = self._read_length
            return self.state()