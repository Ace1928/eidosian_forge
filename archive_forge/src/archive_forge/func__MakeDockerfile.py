from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
import re
import textwrap
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote
def _MakeDockerfile(base_image, main_package, container_workdir, container_home, requirements_path=None, setup_path=None, extra_requirements=None, extra_packages=None, extra_dirs=None):
    """Generates a Dockerfile for building an image.

  It builds on a specified base image to create a container that:
  - installs any dependency specified in a requirements.txt or a setup.py file,
  and any specified dependency packages existing locally or found from PyPI
  - copies all source needed by the main module, and potentially injects an
  entrypoint that, on run, will run that main module

  Args:
    base_image: (str) ID or name of the base image to initialize the build
      stage.
    main_package: (Package) Represents the main application to execute.
    container_workdir: (str) Working directory in the container.
    container_home: (str) $HOME directory in the container.
    requirements_path: (str) Rath of a requirements.txt file.
    setup_path: (str) Path of a setup.py file
    extra_requirements: (List[str]) Required dependencies to install from PyPI.
    extra_packages: (List[str]) User custom dependency packages to install.
    extra_dirs: (List[str]) Directories other than the work_dir required to be
      in the container.

  Returns:
    A string that represents the content of a Dockerfile.
  """
    is_training_prebuilt_image_base = _IsVertexTrainingPrebuiltImage(base_image)
    dockerfile = textwrap.dedent('\n      FROM {base_image}\n      # The directory is created by root. This sets permissions so that any user can\n      # access the folder.\n      RUN mkdir -m 777 -p {workdir} {container_home}\n      WORKDIR {workdir}\n      ENV HOME={container_home}\n\n      # Keeps Python from generating .pyc files in the container\n      ENV PYTHONDONTWRITEBYTECODE=1\n      '.format(base_image=base_image, workdir=shlex_quote(container_workdir), container_home=shlex_quote(container_home)))
    dockerfile += _SitecustomizeRemovalEntry(is_training_prebuilt_image_base)
    dockerfile += _DependencyEntries(is_training_prebuilt_image_base, requirements_path=requirements_path, setup_path=setup_path, extra_requirements=extra_requirements, extra_packages=extra_packages, extra_dirs=extra_dirs)
    dockerfile += _PreparePackageEntry(main_package)
    dockerfile += _GenerateEntrypoint(main_package, is_training_prebuilt_image_base)
    return dockerfile