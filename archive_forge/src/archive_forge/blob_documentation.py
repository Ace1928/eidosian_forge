from mimetypes import guess_type
from . import base
from git.types import Literal

        :return: String describing the mime type of this file (based on the filename)

        :note: Defaults to 'text/plain' in case the actual file type is unknown.
        