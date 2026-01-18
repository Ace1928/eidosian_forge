from io import BytesIO
from .. import osutils
Extract Bazaar metadata from a commit message.

    :param message: Commit message to extract from
    :return: Tuple with original commit message and metadata object
    