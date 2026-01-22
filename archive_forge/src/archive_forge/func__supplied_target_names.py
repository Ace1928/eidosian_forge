import gyp.common
import json
import os
import posixpath
def _supplied_target_names(self):
    return self._additional_compile_target_names | self._test_target_names