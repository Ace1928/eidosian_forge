from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
class OsType(object):
    """The criteria for selecting VM Instances by OS type."""

    class OsShortName(*_StrEnum):
        CENTOS = 'centos'
        DEBIAN = 'debian'
        WINDOWS = 'windows'
        RHEL = 'rhel'
        ROCKY = 'rocky'
        SLES = 'sles'
        SLES_SAP = 'sles-sap'
        UBUNTU = 'ubuntu'

    def __init__(self, short_name, version):
        """Initialize OsType instance.

        Args:
          short_name: str, OS distro name, e.g. 'centos', 'debian'.
          version: str, OS version, e.g. '19.10', '7', '7.8'.
        """
        self.short_name = short_name
        self.version = version

    def __eq__(self, other):
        return self.__dict__ == other.__dict__