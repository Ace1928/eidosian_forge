from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
class DockerInfo:
    """The results of `docker info` and `docker version` for the container runtime."""

    @classmethod
    def init(cls, args: CommonConfig) -> DockerInfo:
        """Initialize and return a DockerInfo instance."""
        command = require_docker().command
        info_stdout = docker_command(args, ['info', '--format', '{{ json . }}'], capture=True, always=True)[0]
        info = json.loads(info_stdout)
        if (server_errors := info.get('ServerErrors')):
            raise ApplicationError('Unable to get container host information: ' + '\n'.join(server_errors))
        version_stdout = docker_command(args, ['version', '--format', '{{ json . }}'], capture=True, always=True)[0]
        version = json.loads(version_stdout)
        info = DockerInfo(args, command, info, version)
        return info

    def __init__(self, args: CommonConfig, engine: str, info: dict[str, t.Any], version: dict[str, t.Any]) -> None:
        self.args = args
        self.engine = engine
        self.info = info
        self.version = version

    @property
    def client(self) -> dict[str, t.Any]:
        """The client version details."""
        client = self.version.get('Client')
        if not client:
            raise ApplicationError('Unable to get container host client information.')
        return client

    @property
    def server(self) -> dict[str, t.Any]:
        """The server version details."""
        server = self.version.get('Server')
        if not server:
            if self.engine == 'podman':
                return self.client
            raise ApplicationError('Unable to get container host server information.')
        return server

    @property
    def client_version(self) -> str:
        """The client version."""
        return self.client['Version']

    @property
    def server_version(self) -> str:
        """The server version."""
        return self.server['Version']

    @property
    def client_major_minor_version(self) -> tuple[int, int]:
        """The client major and minor version."""
        major, minor = self.client_version.split('.')[:2]
        return (int(major), int(minor))

    @property
    def server_major_minor_version(self) -> tuple[int, int]:
        """The server major and minor version."""
        major, minor = self.server_version.split('.')[:2]
        return (int(major), int(minor))

    @property
    def cgroupns_option_supported(self) -> bool:
        """Return True if the `--cgroupns` option is supported, otherwise return False."""
        if self.engine == 'docker':
            return self.client_major_minor_version >= (20, 10) and self.server_major_minor_version >= (20, 10)
        raise NotImplementedError(self.engine)

    @property
    def cgroup_version(self) -> int:
        """The cgroup version of the container host."""
        info = self.info
        host = info.get('host')
        if host:
            return int(host['cgroupVersion'].lstrip('v'))
        try:
            return int(info['CgroupVersion'])
        except KeyError:
            pass
        if self.server_major_minor_version < (20, 10):
            return 1
        message = f'The Docker client version is {self.client_version}. The Docker server version is {self.server_version}. Upgrade your Docker client to version 20.10 or later.'
        if detect_host_properties(self.args).cgroup_v2:
            raise ApplicationError(f'Unsupported Docker client and server combination using cgroup v2. {message}')
        display.warning(f'Detected Docker server cgroup v1 using probing. {message}', unique=True)
        return 1

    @property
    def docker_desktop_wsl2(self) -> bool:
        """Return True if Docker Desktop integrated with WSL2 is detected, otherwise False."""
        info = self.info
        kernel_version = info.get('KernelVersion')
        operating_system = info.get('OperatingSystem')
        dd_wsl2 = kernel_version and kernel_version.endswith('-WSL2') and (operating_system == 'Docker Desktop')
        return dd_wsl2

    @property
    def description(self) -> str:
        """Describe the container runtime."""
        tags = dict(client=self.client_version, server=self.server_version, cgroup=f'v{self.cgroup_version}')
        labels = [self.engine] + [f'{key}={value}' for key, value in tags.items()]
        if self.docker_desktop_wsl2:
            labels.append('DD+WSL2')
        return f'Container runtime: {' '.join(labels)}'