from __future__ import (absolute_import, division, print_function)
def _yum_package_name(name, version, build):
    if version == 'latest':
        return name
    if build == 'latest':
        return '{0}-{1}'.format(name, version)
    return '{0}-{1}-{2}'.format(name, version, build)