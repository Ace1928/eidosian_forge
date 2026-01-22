import codecs
import encodings
class ExtendCodec(codecs.Codec):

    def __init__(self, name, base_encoding, mapping):
        self.name = name
        self.base_encoding = base_encoding
        self.mapping = mapping
        self.reverse = {v: k for k, v in mapping.items()}
        self.max_len = max((len(v) for v in mapping.values()))
        self.info = codecs.CodecInfo(name=self.name, encode=self.encode, decode=self.decode)
        codecs.register_error(name, self.error)

    def _map(self, mapper, output_type, exc_type, input, errors):
        base_error_handler = codecs.lookup_error(errors)
        length = len(input)
        out = output_type()
        while input:
            try:
                part = mapper(input, self.base_encoding, errors=self.name)
                out += part
                break
            except exc_type as e:
                out += mapper(input[:e.start], self.base_encoding, self.name)
                replacement, pos = base_error_handler(e)
                out += replacement
                input = input[pos:]
        return (out, length)

    def encode(self, input, errors='strict'):
        return self._map(codecs.encode, bytes, UnicodeEncodeError, input, errors)

    def decode(self, input, errors='strict'):
        return self._map(codecs.decode, str, UnicodeDecodeError, input, errors)

    def error(self, e):
        if isinstance(e, UnicodeDecodeError):
            for end in range(e.start + 1, e.end + 1):
                s = e.object[e.start:end]
                if s in self.mapping:
                    return (self.mapping[s], end)
        elif isinstance(e, UnicodeEncodeError):
            for end in range(e.start + 1, e.start + self.max_len + 1):
                s = e.object[e.start:end]
                if s in self.reverse:
                    return (self.reverse[s], end)
        e.encoding = self.name
        raise e