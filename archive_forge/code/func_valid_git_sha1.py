from ..errors import InvalidRevisionId
from ..revision import NULL_REVISION
from ..revisionspec import InvalidRevisionSpec, RevisionInfo, RevisionSpec
def valid_git_sha1(hex):
    """Check if `hex` is a validly formatted Git SHA1.

    :param hex: Hex string to validate
    :return: Boolean
    """
    try:
        int(hex, 16)
    except ValueError:
        return False
    else:
        return True