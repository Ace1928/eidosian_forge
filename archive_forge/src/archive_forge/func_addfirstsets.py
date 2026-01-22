import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
def addfirstsets(self) -> None:
    names = list(self.dfas.keys())
    names.sort()
    for name in names:
        if name not in self.first:
            self.calcfirst(name)