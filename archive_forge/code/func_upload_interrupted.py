import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
def upload_interrupted(self):
    if hasattr(self, 'file'):
        temp_location = self.file.temporary_file_path()
        try:
            self.file.close()
            os.remove(temp_location)
        except FileNotFoundError:
            pass