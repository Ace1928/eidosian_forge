from pathlib import Path
from typing import (
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, GetMixin, ListMixin, ObjectDeleteMixin
class GroupPackageManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/packages'
    _obj_cls = GroupPackage
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('exclude_subgroups', 'order_by', 'sort', 'package_type', 'package_name')