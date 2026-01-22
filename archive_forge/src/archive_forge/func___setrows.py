from ..libmp.backend import xrange
import warnings
def __setrows(self, value):
    for key in self.__data.copy():
        if key[0] >= value:
            del self.__data[key]
    self.__rows = value