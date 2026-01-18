import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (

        Tests:
          For components indexed by 0, 1, 2, 3, 4 sets:
            get_index_set_except one, then two (if any) of those sets
            check two items that should be in set_except
            insert item(s) back into these sets via index_getter
        