from collections import namedtuple
import codecs
@staticmethod
def _256_exception_codec(name, exceptions, rexceptions, baseRange=range(32, 256)):
    decoding_map = codecs.make_identity_dict(baseRange)
    decoding_map.update(exceptions)
    encoding_map = codecs.make_encoding_map(decoding_map)
    if rexceptions:
        encoding_map.update(rexceptions)
    return RL_Codecs._makeCodecInfo(name, encoding_map, decoding_map)