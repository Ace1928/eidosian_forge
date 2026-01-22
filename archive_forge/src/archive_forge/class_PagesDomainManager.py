from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ListMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class PagesDomainManager(ListMixin, RESTManager):
    _path = '/pages/domains'
    _obj_cls = PagesDomain