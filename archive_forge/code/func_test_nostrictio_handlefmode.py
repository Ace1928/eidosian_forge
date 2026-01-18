import os
import sys
import string
import random
import dill
def test_nostrictio_handlefmode():
    bench(False, dill.HANDLE_FMODE, False)
    teardown_module()