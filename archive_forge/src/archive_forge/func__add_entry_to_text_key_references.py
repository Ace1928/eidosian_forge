from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def _add_entry_to_text_key_references(self, inv, entry):
    if not self.rich_roots and entry.name == '':
        return
    key = (entry.file_id, entry.revision)
    self.text_key_references.setdefault(key, False)
    if entry.revision == inv.revision_id:
        self.text_key_references[key] = True