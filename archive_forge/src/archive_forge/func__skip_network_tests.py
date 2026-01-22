import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
def _skip_network_tests():
    return os.environ.get('SKLEARN_SKIP_NETWORK_TESTS', '1') == '1'