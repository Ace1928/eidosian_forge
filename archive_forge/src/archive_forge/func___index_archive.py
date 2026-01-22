from pathlib import Path
import sys
def __index_archive(self):
    if self.__fname:
        with open(self.__fname, 'rb') as fp:
            self.__collect_members(fp)
    elif self.__fileobj:
        self.__collect_members(self.__fileobj)
    else:
        raise ArError('Unable to open valid file')