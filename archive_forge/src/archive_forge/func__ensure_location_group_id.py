import os
from datetime import datetime, timezone
from urllib.parse import urljoin
from django.conf import settings
from django.core.files import File, locks
from django.core.files.move import file_move_safe
from django.core.signals import setting_changed
from django.utils._os import safe_join
from django.utils.deconstruct import deconstructible
from django.utils.encoding import filepath_to_uri
from django.utils.functional import cached_property
from .base import Storage
from .mixins import StorageSettingsMixin
def _ensure_location_group_id(self, full_path):
    if os.name == 'posix':
        file_gid = os.stat(full_path).st_gid
        location_gid = os.stat(self.location).st_gid
        if file_gid != location_gid:
            try:
                os.chown(full_path, uid=-1, gid=location_gid)
            except PermissionError:
                pass