from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DockerDaemonValueValuesEnum(_messages.Enum):
    """Optional. Option to specify how (or if) a Docker daemon is provided
    for the build.

    Values:
      DOCKER_DAEMON_UNSPECIFIED: If the option is unspecified, a default will
        be set based on the environment.
      NO_DOCKER: No Docker daemon or functionality will be provided to the
        build.
      NON_PRIVILEGED: A Docker daemon is available during the build that is
        running without privileged mode.
      PRIVILEGED: A Docker daemon will be available that is running in
        privileged mode. This is potentially a security vulnerability and
        should only be used if the user is fully aware of the associated
        risks.
    """
    DOCKER_DAEMON_UNSPECIFIED = 0
    NO_DOCKER = 1
    NON_PRIVILEGED = 2
    PRIVILEGED = 3