import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile
from typing import Optional, List
from distutils.util import rfc822_escape
from . import _normalization, _reqs
from .extern.packaging.markers import Marker
from .extern.packaging.requirements import Requirement
from .extern.packaging.version import Version
from .warnings import SetuptoolsDeprecationWarning
def write_pkg_file(self, file):
    """Write the PKG-INFO format data to a file object."""
    version = self.get_metadata_version()

    def write_field(key, value):
        file.write('%s: %s\n' % (key, value))
    write_field('Metadata-Version', str(version))
    write_field('Name', self.get_name())
    write_field('Version', self.get_version())
    summary = self.get_description()
    if summary:
        write_field('Summary', single_line(summary))
    optional_fields = (('Home-page', 'url'), ('Download-URL', 'download_url'), ('Author', 'author'), ('Author-email', 'author_email'), ('Maintainer', 'maintainer'), ('Maintainer-email', 'maintainer_email'))
    for field, attr in optional_fields:
        attr_val = getattr(self, attr, None)
        if attr_val is not None:
            write_field(field, attr_val)
    license = self.get_license()
    if license:
        write_field('License', rfc822_escape(license))
    for project_url in self.project_urls.items():
        write_field('Project-URL', '%s, %s' % project_url)
    keywords = ','.join(self.get_keywords())
    if keywords:
        write_field('Keywords', keywords)
    platforms = self.get_platforms() or []
    for platform in platforms:
        write_field('Platform', platform)
    self._write_list(file, 'Classifier', self.get_classifiers())
    self._write_list(file, 'Requires', self.get_requires())
    self._write_list(file, 'Provides', self.get_provides())
    self._write_list(file, 'Obsoletes', self.get_obsoletes())
    if hasattr(self, 'python_requires'):
        write_field('Requires-Python', self.python_requires)
    if self.long_description_content_type:
        write_field('Description-Content-Type', self.long_description_content_type)
    self._write_list(file, 'License-File', self.license_files or [])
    _write_requirements(self, file)
    long_description = self.get_long_description()
    if long_description:
        file.write('\n%s' % long_description)
        if not long_description.endswith('\n'):
            file.write('\n')