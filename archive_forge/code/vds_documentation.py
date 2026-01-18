from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
 Return a new low-level dataset identifier for a virtual dataset 