from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def validate_all_flags(self):
    """Verifies whether all flags pass validation.

    Raises:
      AttributeError: Raised if validators work with a non-existing flag.
      IllegalFlagValueError: Raised if validation fails for at least one
          validator.
    """
    all_validators = set()
    for flag in six.itervalues(self._flags()):
        all_validators.update(flag.validators)
    self._assert_validators(all_validators)