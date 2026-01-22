from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from huggingface_hub.utils import parse_datetime
@dataclass
class SpaceVariable:
    """
    Contains information about the current variables of a Space.

    Args:
        key (`str`):
            Variable key. Example: `"MODEL_REPO_ID"`
        value (`str`):
            Variable value. Example: `"the_model_repo_id"`.
        description (`str` or None):
            Description of the variable. Example: `"Model Repo ID of the implemented model"`.
        updatedAt (`datetime` or None):
            datetime of the last update of the variable (if the variable has been updated at least once).
    """
    key: str
    value: str
    description: Optional[str]
    updated_at: Optional[datetime]

    def __init__(self, key: str, values: Dict) -> None:
        self.key = key
        self.value = values['value']
        self.description = values.get('description')
        updated_at = values.get('updatedAt')
        self.updated_at = parse_datetime(updated_at) if updated_at is not None else None