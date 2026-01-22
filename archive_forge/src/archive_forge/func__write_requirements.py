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
def _write_requirements(self, file):
    for req in _reqs.parse(self.install_requires):
        file.write(f'Requires-Dist: {req}\n')
    processed_extras = {}
    for augmented_extra, reqs in self.extras_require.items():
        unsafe_extra, _, condition = augmented_extra.partition(':')
        unsafe_extra = unsafe_extra.strip()
        extra = _normalization.safe_extra(unsafe_extra)
        if extra:
            _write_provides_extra(file, processed_extras, extra, unsafe_extra)
        for req in _reqs.parse_strings(reqs):
            r = _include_extra(req, extra, condition.strip())
            file.write(f'Requires-Dist: {r}\n')
    return processed_extras