from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MachineTypeValueValuesEnum(_messages.Enum):
    """Compute Engine machine type on which to run the build.

    Values:
      UNSPECIFIED: Standard machine type.
      N1_HIGHCPU_8: Highcpu machine with 8 CPUs.
      N1_HIGHCPU_32: Highcpu machine with 32 CPUs.
      E2_HIGHCPU_8: Highcpu e2 machine with 8 CPUs.
      E2_HIGHCPU_32: Highcpu e2 machine with 32 CPUs.
      E2_MEDIUM: E2 machine with 1 CPU.
    """
    UNSPECIFIED = 0
    N1_HIGHCPU_8 = 1
    N1_HIGHCPU_32 = 2
    E2_HIGHCPU_8 = 3
    E2_HIGHCPU_32 = 4
    E2_MEDIUM = 5