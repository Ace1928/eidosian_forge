from mimetypes import guess_type
from . import base
from git.types import Literal
class Blob(base.IndexObject):
    """A Blob encapsulates a git blob object."""
    DEFAULT_MIME_TYPE = 'text/plain'
    type: Literal['blob'] = 'blob'
    executable_mode = 33261
    file_mode = 33188
    link_mode = 40960
    __slots__ = ()

    @property
    def mime_type(self) -> str:
        """
        :return: String describing the mime type of this file (based on the filename)

        :note: Defaults to 'text/plain' in case the actual file type is unknown.
        """
        guesses = None
        if self.path:
            guesses = guess_type(str(self.path))
        return guesses and guesses[0] or self.DEFAULT_MIME_TYPE