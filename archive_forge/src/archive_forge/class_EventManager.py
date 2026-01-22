from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class EventManager(ListMixin, RESTManager):
    _path = '/events'
    _obj_cls = Event
    _list_filters = ('action', 'target_type', 'before', 'after', 'sort', 'scope')