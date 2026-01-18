import sys
import os
import re
import warnings
import types
import unicodedata
def update_all_atts(self, dict_, update_fun=copy_attr_consistent, replace=True, and_source=False):
    """
        Updates all attributes from node or dictionary `dict_`.

        Appends the basic attributes ('ids', 'names', 'classes',
        'dupnames', but not 'source') and then, for all other attributes in
        dict_, updates the same attribute in self.  When attributes with the
        same identifier appear in both self and dict_, the two values are
        merged based on the value of update_fun.  Generally, when replace is
        True, the values in self are replaced or merged with the values in
        dict_; otherwise, the values in self may be preserved or merged.  When
        and_source is True, the 'source' attribute is included in the copy.

        NOTE: When replace is False, and self contains a 'source' attribute,
              'source' is not replaced even when dict_ has a 'source'
              attribute, though it may still be merged into a list depending
              on the value of update_fun.
        NOTE: It is easier to call the update-specific methods then to pass
              the update_fun method to this function.
        """
    if isinstance(dict_, Node):
        dict_ = dict_.attributes
    if and_source:
        filter_fun = self.is_not_list_attribute
    else:
        filter_fun = self.is_not_known_attribute
    self.update_basic_atts(dict_)
    for att in filter(filter_fun, dict_):
        update_fun(self, att, dict_[att], replace)