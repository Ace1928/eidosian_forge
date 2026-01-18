import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
Assert that _url_scheme_re correctly matches

            :param scheme_and_path: The (scheme, path) that should be matched
                can be None, to indicate it should not match
            