from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
Initialize OsType instance.

        Args:
          short_name: str, OS distro name, e.g. 'centos', 'debian'.
          version: str, OS version, e.g. '19.10', '7', '7.8'.
        