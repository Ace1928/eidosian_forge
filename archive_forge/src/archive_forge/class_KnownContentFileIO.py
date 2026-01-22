import os
from parso import file_io
class KnownContentFileIO(file_io.KnownContentFileIO, FileIOFolderMixin):
    pass