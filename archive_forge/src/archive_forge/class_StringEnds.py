import abc
import copy
from neutron_lib import exceptions
class StringEnds(StringMatchingFilterObj):

    def __init__(self, matching_string):
        super().__init__()
        self.ends = matching_string

    def filter(self, column):
        return column.endswith(self.ends)