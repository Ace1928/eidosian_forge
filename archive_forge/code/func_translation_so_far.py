import warnings
from collections import defaultdict
from math import log
def translation_so_far(self):
    translation = []
    self.__build_translation(self, translation)
    return translation