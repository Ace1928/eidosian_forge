import os
from looseversion import LooseVersion
from .info import URL as __url__, STATUS as __status__, __version__
from .utils.config import NipypeConfig
from .utils.logger import Logging
from .refs import due
from .pkg_info import get_pkg_info as _get_pkg_info
from .pipeline import Node, MapNode, JoinNode, Workflow
from .interfaces import (
class NipypeTester(object):

    def __call__(self, doctests=True, parallel=False):
        try:
            import pytest
        except ImportError:
            raise RuntimeError('py.test not installed, run: pip install pytest')
        args = []
        if not doctests:
            args.extend(['-p', 'no:doctest'])
        if parallel:
            try:
                import xdist
            except ImportError:
                raise RuntimeError('pytest-xdist required for parallel run')
            args.append('-n auto')
        args.append(os.path.dirname(__file__))
        pytest.main(args=args)