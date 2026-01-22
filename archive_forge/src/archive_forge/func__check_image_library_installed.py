import datetime
import posixpath
from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData
from django.utils.translation import gettext_lazy as _
def _check_image_library_installed(self):
    try:
        from PIL import Image
    except ImportError:
        return [checks.Error('Cannot use ImageField because Pillow is not installed.', hint='Get Pillow at https://pypi.org/project/Pillow/ or run command "python -m pip install Pillow".', obj=self, id='fields.E210')]
    else:
        return []