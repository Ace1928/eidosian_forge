import os
from parso import file_io
class FileIOFolderMixin:

    def get_parent_folder(self):
        return FolderIO(os.path.dirname(self.path))