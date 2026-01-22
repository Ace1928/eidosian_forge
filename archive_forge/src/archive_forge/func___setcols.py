from ..libmp.backend import xrange
import warnings
def __setcols(self, value):
    for key in self.__data.copy():
        if key[1] >= value:
            del self.__data[key]
    self.__cols = value