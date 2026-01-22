from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
class BulkImportManager(CreateMixin, RetrieveMixin, RESTManager):
    _path = '/bulk_imports'
    _obj_cls = BulkImport
    _create_attrs = RequiredOptional(required=('configuration', 'entities'))
    _list_filters = ('sort', 'status')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> BulkImport:
        return cast(BulkImport, super().get(id=id, lazy=lazy, **kwargs))