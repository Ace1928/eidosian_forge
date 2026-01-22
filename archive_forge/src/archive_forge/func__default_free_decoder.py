import functools
import re
from ovs.flow.decoders import decode_default
@staticmethod
def _default_free_decoder(key):
    """Default decoder for free keywords."""
    return (key, True)