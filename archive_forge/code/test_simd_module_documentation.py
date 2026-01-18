import pytest
from numpy.core._simd import targets

This testing unit only for checking the sanity of common functionality,
therefore all we need is just to take one submodule that represents any
of enabled SIMD extensions to run the test on it and the second submodule
required to run only one check related to the possibility of mixing
the data types among each submodule.
