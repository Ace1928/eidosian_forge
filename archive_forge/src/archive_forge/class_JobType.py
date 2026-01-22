from enum import Enum
from typing import Any, Dict, Optional
from ray._private.pydantic_compat import BaseModel, Field, PYDANTIC_INSTALLED
from ray.dashboard.modules.job.common import JobStatus
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class JobType(str, Enum):
    """An enumeration for describing the different job types.

        NOTE:
            This field is still experimental and may change in the future.
        """
    SUBMISSION = 'SUBMISSION'
    DRIVER = 'DRIVER'