from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
def _add_opt_variable(opt_name_v2, var_name_v1, idx, suffix_v2, reader, var_map, var_names_map):
    """Adds a new optimizer v2 variable."""
    var_name_v2 = 'training/' + opt_name_v2 + '/' + var_name_v1[:idx] + suffix_v2
    tensor = reader.get_tensor(var_name_v1)
    _add_new_variable(tensor, var_name_v2, var_name_v1, var_map, var_names_map)