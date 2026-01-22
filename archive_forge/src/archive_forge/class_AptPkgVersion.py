import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
class AptPkgVersion(BaseVersion):
    """Represents a Debian package version, using apt_pkg.VersionCompare"""

    def __init__(self, version):
        if not _have_apt_pkg:
            raise NotImplementedError('apt_pkg not available; install the python-apt package')
        super(AptPkgVersion, self).__init__(version)

    def _compare(self, other):
        return apt_pkg.version_compare(str(self), str(other))