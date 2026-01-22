import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_feature_metadata(self):
    """
        Build the features components of the User-Agent header string.

        Botocore currently does not report any features. This may change in a
        future version.
        """
    return []