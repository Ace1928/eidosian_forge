from __future__ import annotations
import logging # isort:skip
import re
from types import ModuleType
from ...core.types import PathLike
from ...util.dependencies import import_required
from .code import CodeHandler
class StripMagicsProcessor(nbconvert.preprocessors.Preprocessor):
    """
            Preprocessor to convert notebooks to Python source while stripping
            out all magics (i.e IPython specific syntax).
            """
    _magic_pattern = re.compile('^\\s*(?P<magic>%%\\w\\w+)($|(\\s+))')

    def strip_magics(self, source: str) -> str:
        """
                Given the source of a cell, filter out all cell and line magics.
                """
        filtered: list[str] = []
        for line in source.splitlines():
            match = self._magic_pattern.match(line)
            if match is None:
                filtered.append(line)
            else:
                msg = 'Stripping out IPython magic {magic} in code cell {cell}'
                message = msg.format(cell=self._cell_counter, magic=match.group('magic'))
                log.warning(message)
        return '\n'.join(filtered)

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            self._cell_counter += 1
            cell['source'] = self.strip_magics(cell['source'])
        return (cell, resources)

    def __call__(self, nb, resources):
        self._cell_counter = 0
        return self.preprocess(nb, resources)