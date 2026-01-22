from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
@dataclass
class MultipartUploadPart:
    part_number: int
    etag: str
    url: Optional[str] = None

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.part_number, proto.etag or None, proto.url or None)

    def to_dict(self):
        return {'part_number': self.part_number, 'etag': self.etag, 'url': self.url}