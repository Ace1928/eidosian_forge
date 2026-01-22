from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
@dataclass
class CreateMultipartUploadResponse:
    upload_id: Optional[str]
    credentials: List[MultipartUploadCredential]

    def to_proto(self):
        response = ProtoCreateMultipartUpload.Response()
        if self.upload_id:
            response.upload_id = self.upload_id
        response.credentials.extend([credential.to_proto() for credential in self.credentials])
        return response

    @classmethod
    def from_dict(cls, dict_):
        credentials = [MultipartUploadCredential.from_dict(cred) for cred in dict_['credentials']]
        return cls(upload_id=dict_.get('upload_id'), credentials=credentials)