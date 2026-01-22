from IPython.utils.dir2 import dir2
import pytest
class SillierWithDir(MisbehavingGetattr):

    def __dir__(self):
        return ['some_method']