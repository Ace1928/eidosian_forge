import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
class BaseVersion(object):
    """Base class for classes representing Debian versions

    It doesn't implement any comparison, but it does check for valid versions
    according to Section 5.6.12 in the Debian Policy Manual.  Since splitting
    the version into epoch, upstream_version, and debian_revision components is
    pretty much free with the validation, it sets those fields as properties of
    the object, and sets the raw version to the full_version property.  A
    missing epoch or debian_revision results in the respective property set to
    None.  Setting any of the properties results in the full_version being
    recomputed and the rest of the properties set from that.

    It also implements __str__, just returning the raw version given to the
    initializer.
    """
    re_valid_version = re.compile('^((?P<epoch>\\d+):)?(?P<upstream_version>[A-Za-z0-9.+:~-]+?)(-(?P<debian_revision>[A-Za-z0-9+.~]+))?$')
    magic_attrs = ('full_version', 'epoch', 'upstream_version', 'debian_revision', 'debian_version')

    def __init__(self, version):
        if isinstance(version, BaseVersion):
            version = str(version)
        self.full_version = version

    def _set_full_version(self, version):
        m = self.re_valid_version.match(version)
        if not m:
            raise ValueError('Invalid version string %r' % version)
        if m.group('epoch') is None and ':' in m.group('upstream_version'):
            raise ValueError('Invalid version string %r' % version)
        self.__full_version = version
        self.__epoch = m.group('epoch')
        self.__upstream_version = m.group('upstream_version')
        self.__debian_revision = m.group('debian_revision')

    def __setattr__(self, attr, value):
        if attr not in self.magic_attrs:
            super(BaseVersion, self).__setattr__(attr, value)
            return
        if attr == 'debian_version':
            attr = 'debian_revision'
        if attr == 'full_version':
            self._set_full_version(str(value))
        else:
            if value is not None:
                value = str(value)
            private = '_BaseVersion__%s' % attr
            old_value = getattr(self, private)
            setattr(self, private, value)
            try:
                self._update_full_version()
            except ValueError:
                setattr(self, private, old_value)
                self._update_full_version()
                raise ValueError('Setting %s to %r results in invalid version' % (attr, value))

    def __getattr__(self, attr):
        if attr not in self.magic_attrs:
            return super(BaseVersion, self).__getattribute__(attr)
        if attr == 'debian_version':
            attr = 'debian_revision'
        private = '_BaseVersion__%s' % attr
        return getattr(self, private)

    def _update_full_version(self):
        version = ''
        if self.__epoch is not None:
            version += self.__epoch + ':'
        version += self.__upstream_version
        if self.__debian_revision:
            version += '-' + self.__debian_revision
        self.full_version = version

    def __str__(self):
        return self.full_version if self.full_version is not None else ''

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self)

    def _compare(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __hash__(self):
        return hash(str(self))