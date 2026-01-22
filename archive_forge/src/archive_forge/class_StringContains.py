import abc
import copy
from neutron_lib import exceptions
class StringContains(StringMatchingFilterObj):

    def __init__(self, matching_string):
        super().__init__()
        self.contains = matching_string

    def filter(self, column):
        return column.contains(self.contains)