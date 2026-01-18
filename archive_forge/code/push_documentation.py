from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
Import a revision into this Git repository.

        :param revid: Revision id of the revision
        :param roundtrip: Whether to roundtrip bzr metadata
        