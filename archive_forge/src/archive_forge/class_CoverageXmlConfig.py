from __future__ import annotations
import os
import time
from xml.etree.ElementTree import (
from xml.dom import (
from ...io import (
from ...util_common import (
from ...util import (
from ...data import (
from ...provisioning import (
from .combine import (
from . import (
class CoverageXmlConfig(CoverageCombineConfig):
    """Configuration for the coverage xml command."""