import os
import re
import sys
from typing import List, Optional
from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import (
def _egg_link_names(raw_name: str) -> List[str]:
    """
    Convert a Name metadata value to a .egg-link name, by applying
    the same substitution as pkg_resources's safe_name function.
    Note: we cannot use canonicalize_name because it has a different logic.

    We also look for the raw name (without normalization) as setuptools 69 changed
    the way it names .egg-link files (https://github.com/pypa/setuptools/issues/4167).
    """
    return [re.sub('[^A-Za-z0-9.]+', '-', raw_name) + '.egg-link', f'{raw_name}.egg-link']