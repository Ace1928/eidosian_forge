from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectMergeRequestDraftNote(ObjectDeleteMixin, SaveMixin, RESTObject):

    def publish(self, **kwargs: Any) -> None:
        path = f'{self.manager.path}/{self.encoded_id}/publish'
        self.manager.gitlab.http_put(path, **kwargs)