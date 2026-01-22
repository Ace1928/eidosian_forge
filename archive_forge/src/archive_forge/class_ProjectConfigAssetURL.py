from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from wasabi import msg
class ProjectConfigAssetURL(BaseModel):
    dest: StrictStr = Field(..., title='Destination of downloaded asset')
    url: Optional[StrictStr] = Field(None, title='URL of asset')
    checksum: Optional[str] = Field(None, title='MD5 hash of file', regex='([a-fA-F\\d]{32})')
    description: StrictStr = Field('', title='Description of asset')