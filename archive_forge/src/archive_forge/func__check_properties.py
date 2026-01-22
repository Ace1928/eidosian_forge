from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def _check_properties(self):
    """Verify that all revision properties are OK."""
    for name, value in self.properties.items():
        not_text = not isinstance(name, str)
        if not_text or osutils.contains_whitespace(name):
            raise ValueError('invalid property name %r' % name)
        if not isinstance(value, (str, bytes)):
            raise ValueError('invalid property value %r for %r' % (value, name))