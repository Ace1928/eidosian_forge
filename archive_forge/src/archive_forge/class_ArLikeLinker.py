from __future__ import annotations
import enum
import typing as T
class ArLikeLinker:
    std_args = ['-csr']

    def can_linker_accept_rsp(self) -> bool:
        return False

    def get_std_link_args(self, env: 'Environment', is_thin: bool) -> T.List[str]:
        return self.std_args

    def get_output_args(self, target: str) -> T.List[str]:
        return [target]

    def rsp_file_syntax(self) -> RSPFileSyntax:
        return RSPFileSyntax.GCC