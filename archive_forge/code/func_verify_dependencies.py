import gast as ast
import os
import re
from time import time
def verify_dependencies(self):
    """
        Checks no analysis are called before a transformation,
        as the transformation could invalidate the analysis.
        """
    for i in range(1, len(self.deps)):
        assert not (isinstance(self.deps[i], Transformation) and isinstance(self.deps[i - 1], Analysis)), 'invalid dep order for %s' % self