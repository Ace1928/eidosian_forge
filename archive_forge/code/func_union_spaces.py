import collections
from minerl.herobraine.hero.handler import Handler
import numpy as np
from functools import reduce
from typing import List, Tuple
from minerl.herobraine.hero.spaces import Box, Dict, Enum, MineRLSpace
def union_spaces(hdls_1: List[Handler], hdls_2: List[Handler]) -> List[MineRLSpace]:
    hdls = hdls_1 + hdls_2
    hdl_dict = collections.defaultdict(list)
    _ = [hdl_dict[hdl.to_string()].append(hdl) for hdl in hdls]
    merged_hdls = [reduce(lambda a, b: a | b, matching) for matching in hdl_dict.values()]
    return merged_hdls