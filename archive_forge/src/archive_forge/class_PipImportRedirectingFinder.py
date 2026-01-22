import sys
import runpy  # noqa: E402
from importlib.machinery import PathFinder  # noqa: E402
from os.path import dirname  # noqa: E402
class PipImportRedirectingFinder:

    @classmethod
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'pip':
            return None
        spec = PathFinder.find_spec(fullname, [PIP_SOURCES_ROOT], target)
        assert spec, (PIP_SOURCES_ROOT, fullname)
        return spec