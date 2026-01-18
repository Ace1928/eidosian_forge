import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
Generates a JobConfig object from json.

        Examples:
            .. testcode::

                from ray.job_config import JobConfig

                job_config = JobConfig.from_json(
                    {"runtime_env": {"working_dir": "uri://abc"}})

        Args:
            job_config_json: The job config json dictionary.
        