from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def unregister_transformer(self, transformer):
    """Unregister a transformer instance."""
    if transformer in self._transformers:
        self._transformers.remove(transformer)