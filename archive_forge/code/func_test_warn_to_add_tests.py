import numpy as np
import pytest
from unittest.mock import Mock, patch
def test_warn_to_add_tests(self):
    assert len(np.__config__.DisplayModes) == 2, 'New mode detected, please add UT if applicable and increment this count'