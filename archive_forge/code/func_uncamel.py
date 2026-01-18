import gast as ast
import os
import re
from time import time
def uncamel(name):
    """Transform CamelCase naming convention into C-ish convention."""
    s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()