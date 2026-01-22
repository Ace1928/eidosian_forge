from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import six
Runs a single check and returns the result and an optional fix.

    Returns:
      A tuple of two elements. The first element should have the same attributes
      as a check_base.Result object. The second element should either be a fixer
      function that can used to fix an error (indicated by the "passed"
      attribute being False in the first element), or None if the check passed
      or if it failed with no applicable fix. If there is a fixer function it is
      assumed that calling it will return True if it makes changes that warrant
      running a check again.
    