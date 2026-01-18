import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

        Fixture for an indexer to pass to obj.loc to get/set the full length of the
        object.

        In some cases, assumes that obj.index is the default RangeIndex.
        