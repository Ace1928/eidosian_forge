from collections import namedtuple
import codecs
class Codec(codecs.Codec):

    def encode(self, input, errors='strict', charmap_encode=codecs.charmap_encode, encoding_map=encoding_map):
        return charmap_encode(input, errors, encoding_map)

    def decode(self, input, errors='strict', charmap_decode=codecs.charmap_decode, decoding_map=decoding_map):
        return charmap_decode(input, errors, decoding_map)