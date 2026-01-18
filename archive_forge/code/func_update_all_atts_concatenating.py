import sys
import os
import re
import warnings
import types
import unicodedata
def update_all_atts_concatenating(self, dict_, replace=True, and_source=False):
    """
        Updates all attributes from node or dictionary `dict_`.

        Appends the basic attributes ('ids', 'names', 'classes',
        'dupnames', but not 'source') and then, for all other attributes in
        dict_, updates the same attribute in self.  When attributes with the
        same identifier appear in both self and dict_ whose values aren't each
        lists and replace is True, the values in self are replaced with the
        values in dict_; if the values from self and dict_ for the given
        identifier are both of list type, then the two lists are concatenated
        and the result stored in self; otherwise, the values in self are
        preserved.  When and_source is True, the 'source' attribute is
        included in the copy.

        NOTE: When replace is False, and self contains a 'source' attribute,
              'source' is not replaced even when dict_ has a 'source'
              attribute, though it may still be merged into a list depending
              on the value of update_fun.
        """
    self.update_all_atts(dict_, Element.copy_attr_concatenate, replace, and_source)