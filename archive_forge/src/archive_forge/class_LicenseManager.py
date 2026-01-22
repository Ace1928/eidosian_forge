from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class LicenseManager(RetrieveMixin, RESTManager):
    _path = '/templates/licenses'
    _obj_cls = License
    _list_filters = ('popular',)
    _optional_get_attrs = ('project', 'fullname')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> License:
        return cast(License, super().get(id=id, lazy=lazy, **kwargs))