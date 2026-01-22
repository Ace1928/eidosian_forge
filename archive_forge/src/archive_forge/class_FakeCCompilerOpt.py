import unittest
from os import sys, path
class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ('arch', 'compiler', 'extra_args')

    def __init__(self, *args, **kwargs):
        CCompilerOpt.__init__(self, None, **kwargs)

    def dist_compile(self, sources, flags, **kwargs):
        return sources

    def dist_info(self):
        return FakeCCompilerOpt.fake_info

    @staticmethod
    def dist_log(*args, stderr=False):
        pass