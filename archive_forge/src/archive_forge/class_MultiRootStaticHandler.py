from __future__ import annotations
import logging # isort:skip
import os
from pathlib import Path
from tornado.web import HTTPError, StaticFileHandler
from ...core.types import PathLike
class MultiRootStaticHandler(StaticFileHandler):

    def initialize(self, root: dict[str, PathLike]) -> None:
        self.root = root
        self.default_filename = None

    @classmethod
    def get_absolute_path(cls, root: dict[str, PathLike], path: str) -> str:
        try:
            name, artifact_path = path.split(os.sep, 1)
        except ValueError:
            raise HTTPError(404)
        artifacts_dir = root.get(name, None)
        if artifacts_dir is not None:
            return super().get_absolute_path(str(artifacts_dir), artifact_path)
        else:
            raise HTTPError(404)

    def validate_absolute_path(self, root: dict[str, PathLike], absolute_path: str) -> str | None:
        for artifacts_dir in root.values():
            if Path(absolute_path).is_relative_to(artifacts_dir):
                return super().validate_absolute_path(str(artifacts_dir), absolute_path)
        return None