from __future__ import annotations
from .backends import Backend
from .. import mlog
from ..mesonlib import MesonBugException
class NoneBackend(Backend):
    name = 'none'

    def generate(self, capture: bool=False, vslite_ctx: dict=None) -> None:
        if capture:
            raise MesonBugException("We do not expect the none backend to generate with 'capture = True'")
        if vslite_ctx:
            raise MesonBugException("We do not expect the none backend to be given a valid 'vslite_ctx'")
        if self.build.get_targets():
            raise MesonBugException('None backend cannot generate target rules, but should have failed earlier.')
        mlog.log('Generating simple install-only backend')
        self.serialize_tests()
        self.create_install_data_files()