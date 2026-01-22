import re
import sys
class _Tool:
    """Represents a tool used by MSVS or MSBuild.

  Attributes:
      msvs_name: The name of the tool in MSVS.
      msbuild_name: The name of the tool in MSBuild.
  """

    def __init__(self, msvs_name, msbuild_name):
        self.msvs_name = msvs_name
        self.msbuild_name = msbuild_name