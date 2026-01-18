from enum import Enum
from typing import Any, Dict, Optional
from ray._private.pydantic_compat import BaseModel, Field, PYDANTIC_INSTALLED
from ray.dashboard.modules.job.common import JobStatus
from ray.util.annotations import PublicAPI

        Job data with extra details about its driver and its submission.
        