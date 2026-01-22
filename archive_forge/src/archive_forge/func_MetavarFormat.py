from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MetavarFormat(arg_name):
    """Gets arg name in upper snake case."""
    return SnakeCase(arg_name.lstrip('-')).upper()