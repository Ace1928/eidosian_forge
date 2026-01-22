from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from wasabi import msg
class ProjectConfigAssetGitItem(BaseModel):
    repo: StrictStr = Field(..., title='URL of Git repo to download from')
    path: StrictStr = Field(..., title='File path or sub-directory to download (used for sparse checkout)')
    branch: StrictStr = Field('master', title='Branch to clone from')