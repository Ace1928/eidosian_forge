from __future__ import annotations
import sys
import math
def _check_api_version(api_version):
    if api_version is not None and api_version != '2021.12':
        raise ValueError('Only the 2021.12 version of the array API specification is currently supported')