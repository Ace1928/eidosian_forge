import re
import math
def __ensure_cache(self, column):
    while column >= len(self.__cache):
        self.__add_column()