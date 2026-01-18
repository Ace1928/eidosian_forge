from __future__ import (absolute_import, division, print_function)
import re
def validate_result(result, desc):
    if not result:
        raise AssertionError('failed on test ' + desc)