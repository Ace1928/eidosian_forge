import os
import subprocess
import sys

    Test that import joblib works even if _multiprocessing is missing.

    pytest has already imported everything from joblib. The most reasonable way
    to test importing joblib with modified environment is to invoke a separate
    Python process. This also ensures that we don't break other tests by
    importing a bad `_multiprocessing` module.
    