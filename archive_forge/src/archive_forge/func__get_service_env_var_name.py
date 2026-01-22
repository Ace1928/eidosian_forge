import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_service_env_var_name(self):
    transformed_service_id_env = self._transformed_service_id.upper()
    return f'AWS_ENDPOINT_URL_{transformed_service_id_env}'