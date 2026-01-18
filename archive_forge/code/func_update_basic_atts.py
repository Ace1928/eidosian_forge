import sys
import os
import re
import warnings
import types
import unicodedata
def update_basic_atts(self, dict_):
    """
        Update basic attributes ('ids', 'names', 'classes',
        'dupnames', but not 'source') from node or dictionary `dict_`.
        """
    if isinstance(dict_, Node):
        dict_ = dict_.attributes
    for att in self.basic_attributes:
        self.append_attr_list(att, dict_.get(att, []))