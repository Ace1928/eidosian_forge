import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
Verify sleep count is handled properly