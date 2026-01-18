import logging
from typing import Tuple
from wandb.docker import is_docker_installed
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import docker_image_exists
from .abstract import AbstractRegistry
Check if an image exists in the local registry.

        Arguments:
            image_uri (str): The uri of the image.

        Returns:
            bool: True.
        