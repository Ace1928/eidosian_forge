from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators_classes
def validate_mutual_exclusion(flags_dict):
    flag_count = sum((1 for val in flags_dict.values() if val is not None))
    if flag_count == 1 or (not required and flag_count == 0):
        return True
    raise _exceptions.ValidationError('{} one of ({}) must have a value other than None.'.format('Exactly' if required else 'At most', ', '.join(flag_names)))