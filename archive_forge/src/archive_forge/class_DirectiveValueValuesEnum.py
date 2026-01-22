from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectiveValueValuesEnum(_messages.Enum):
    """Required. The recovered Dockerfile directive used to construct this
    layer.

    Values:
      DIRECTIVE_UNSPECIFIED: Default value for unsupported/missing directive.
      MAINTAINER: https://docs.docker.com/engine/reference/builder/
      RUN: https://docs.docker.com/engine/reference/builder/
      CMD: https://docs.docker.com/engine/reference/builder/
      LABEL: https://docs.docker.com/engine/reference/builder/
      EXPOSE: https://docs.docker.com/engine/reference/builder/
      ENV: https://docs.docker.com/engine/reference/builder/
      ADD: https://docs.docker.com/engine/reference/builder/
      COPY: https://docs.docker.com/engine/reference/builder/
      ENTRYPOINT: https://docs.docker.com/engine/reference/builder/
      VOLUME: https://docs.docker.com/engine/reference/builder/
      USER: https://docs.docker.com/engine/reference/builder/
      WORKDIR: https://docs.docker.com/engine/reference/builder/
      ARG: https://docs.docker.com/engine/reference/builder/
      ONBUILD: https://docs.docker.com/engine/reference/builder/
      STOPSIGNAL: https://docs.docker.com/engine/reference/builder/
      HEALTHCHECK: https://docs.docker.com/engine/reference/builder/
      SHELL: https://docs.docker.com/engine/reference/builder/
    """
    DIRECTIVE_UNSPECIFIED = 0
    MAINTAINER = 1
    RUN = 2
    CMD = 3
    LABEL = 4
    EXPOSE = 5
    ENV = 6
    ADD = 7
    COPY = 8
    ENTRYPOINT = 9
    VOLUME = 10
    USER = 11
    WORKDIR = 12
    ARG = 13
    ONBUILD = 14
    STOPSIGNAL = 15
    HEALTHCHECK = 16
    SHELL = 17