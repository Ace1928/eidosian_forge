from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def validate_boolean_mutual_exclusion(flags_dict):
    flag_count = sum((bool(val) for val in flags_dict.values()))
    if flag_count == 1 or (not required and flag_count == 0):
        return True
    raise _exceptions.ValidationError('{} one of ({}) must be True.'.format('Exactly' if required else 'At most', ', '.join(flag_names)))