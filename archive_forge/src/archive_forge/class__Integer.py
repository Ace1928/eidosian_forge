import re
import sys
class _Integer(_Type):
    """Integer settings."""

    def __init__(self, msbuild_base=10):
        _Type.__init__(self)
        self._msbuild_base = msbuild_base

    def ValidateMSVS(self, value):
        self.ConvertToMSBuild(value)

    def ValidateMSBuild(self, value):
        int(value, self._msbuild_base)

    def ConvertToMSBuild(self, value):
        msbuild_format = self._msbuild_base == 10 and '%d' or '0x%04x'
        return msbuild_format % int(value)