import pytest
import numpy as np
import ase.io.ulm as ulm
class MyFile:

    def __fspath__(self):
        return 'hello'