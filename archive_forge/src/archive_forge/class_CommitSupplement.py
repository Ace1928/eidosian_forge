from io import BytesIO
from .. import osutils
class CommitSupplement:
    """Supplement for a Bazaar revision roundtripped into Git.

    :ivar revision_id: Revision id, as string
    :ivar properties: Revision properties, as dictionary
    :ivar explicit_parent_ids: Parent ids (needed if there are ghosts)
    :ivar verifiers: Verifier information
    """
    revision_id = None
    explicit_parent_ids = None

    def __init__(self):
        self.properties = {}
        self.verifiers = {}

    def __nonzero__(self):
        return bool(self.revision_id or self.properties or self.explicit_parent_ids)