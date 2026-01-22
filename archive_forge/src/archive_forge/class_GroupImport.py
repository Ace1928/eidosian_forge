from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, DownloadMixin, GetWithoutIdMixin, RefreshMixin
from gitlab.types import RequiredOptional
class GroupImport(RESTObject):
    _id_attr = None