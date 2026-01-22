from pathlib import Path
from typing import Any, Dict
from ray.autoscaler._private.cli_logger import cli_logger
def _check_helper(cname, template, docker_cmd):
    return ' '.join([docker_cmd, 'inspect', '-f', "'{{" + template + "}}'", cname, '||', 'true'])