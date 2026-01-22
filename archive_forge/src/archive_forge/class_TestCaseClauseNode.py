from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
@dataclass(unsafe_hash=True)
class TestCaseClauseNode(BaseNode):
    testcase: SymbolNode
    condition: BaseNode
    block: CodeBlockNode
    endtestcase: SymbolNode

    def __init__(self, testcase: SymbolNode, condition: BaseNode, block: CodeBlockNode, endtestcase: SymbolNode):
        super().__init__(condition.lineno, condition.colno, condition.filename)
        self.testcase = testcase
        self.condition = condition
        self.block = block
        self.endtestcase = endtestcase