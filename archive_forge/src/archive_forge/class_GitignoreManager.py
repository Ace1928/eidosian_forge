from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
class GitignoreManager(RetrieveMixin, RESTManager):
    _path = '/templates/gitignores'
    _obj_cls = Gitignore

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Gitignore:
        return cast(Gitignore, super().get(id=id, lazy=lazy, **kwargs))