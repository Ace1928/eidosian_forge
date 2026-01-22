import abc
import copy
from neutron_lib import exceptions
class StringStarts(StringMatchingFilterObj):

    def __init__(self, matching_string):
        super().__init__()
        self.starts = matching_string

    def filter(self, column):
        return column.startswith(self.starts)