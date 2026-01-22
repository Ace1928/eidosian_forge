from typing import List, Literal, Optional
from pydantic import BaseModel
class WebhookPayloadRepo(ObjectId):
    owner: ObjectId
    head_sha: Optional[str] = None
    name: str
    private: bool
    subdomain: Optional[str] = None
    tags: Optional[List[str]] = None
    type: Literal['dataset', 'model', 'space']
    url: WebhookPayloadUrl