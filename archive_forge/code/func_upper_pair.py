from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def upper_pair(self):
    first, second, even_over = self
    return (0, 2) if bool(first % 2) ^ even_over else (1, 3)