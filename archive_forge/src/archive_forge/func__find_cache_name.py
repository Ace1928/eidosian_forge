from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _find_cache_name(self, default='default'):
    cache_file = next(self._cache_dir_contents(), None)
    cache_name = default
    if cache_file:
        cache_name = os.path.basename(cache_file)[:-len(self.cache_extension)]
    return cache_name