import numpy as np
from functools import reduce
from collections import OrderedDict
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.util import union_spaces, flatten_spaces, intersect_space
from minerl.herobraine.wrapper import EnvWrapper

    Normalizes and flattens a typical env space for obfuscation.
    common_envs : specified
    