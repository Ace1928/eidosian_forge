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
def _write_provides_extra(file, processed_extras, safe, unsafe):
    previous = processed_extras.get(safe)
    if previous == unsafe:
        SetuptoolsDeprecationWarning.emit('Ambiguity during "extra" normalization for dependencies.', f'\n            {previous!r} and {unsafe!r} normalize to the same value:\n\n                {safe!r}\n\n            In future versions, setuptools might halt the build process.\n            ', see_url='https://peps.python.org/pep-0685/')
    else:
        processed_extras[safe] = unsafe
        file.write(f'Provides-Extra: {safe}\n')