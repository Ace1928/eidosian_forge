from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _utf8_decode(self, field, value):
    try:
        return value.decode('utf-8')
    except UnicodeDecodeError:
        self.warning('%s not in utf8 - replacing unknown characters' % (field,))
        return value.decode('utf-8', 'replace')