from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _legal_revision_properties(self, props):
    """Clean-up any revision properties we can't handle."""
    result = {}
    if props is not None:
        for name, value in props.items():
            if value is None:
                self.warning('converting None to empty string for property %s' % (name,))
                result[name] = ''
            else:
                result[name] = value
    return result