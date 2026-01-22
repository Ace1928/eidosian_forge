from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
class BulkImportEntityManager(RetrieveMixin, RESTManager):
    _path = '/bulk_imports/{bulk_import_id}/entities'
    _obj_cls = BulkImportEntity
    _from_parent_attrs = {'bulk_import_id': 'id'}
    _list_filters = ('sort', 'status')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> BulkImportEntity:
        return cast(BulkImportEntity, super().get(id=id, lazy=lazy, **kwargs))