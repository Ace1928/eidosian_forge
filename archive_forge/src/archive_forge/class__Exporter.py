import inspect
from collections import defaultdict
from textwrap import dedent
from typing import Any, ClassVar, Dict, List, Tuple, Type
import numpy as np
class _Exporter(type):
    exports: ClassVar[Dict[str, List[Tuple[str, str]]]] = defaultdict(list)

    def __init__(cls, name: str, bases: Tuple[Type[Any], ...], dct: Dict[str, Any]) -> None:
        for k, v in dct.items():
            if k.startswith('export'):
                if not isinstance(v, staticmethod):
                    raise ValueError('Only staticmethods could be named as export.*')
                export = getattr(cls, k)
                Snippets[name].append(process_snippet(name, k, export))
                np.random.seed(seed=0)
                export()
        super().__init__(name, bases, dct)