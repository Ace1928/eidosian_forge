import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
def _parse_function(self, func: Callable, params_re: str='.*', return_re: str='.*') -> Tuple[bool, IndexedOrderedDict[str, 'AnnotatedParam'], 'AnnotatedParam']:
    sig = inspect.signature(func)
    annotations = get_type_hints(func)
    res: IndexedOrderedDict[str, 'AnnotatedParam'] = IndexedOrderedDict()
    class_method = False
    for k, w in sig.parameters.items():
        if k == 'self':
            res[k] = SelfParam(w)
            class_method = True
        else:
            anno = annotations.get(k, w.annotation)
            res[k] = self.__class__.parse_annotation(anno, w)
    anno = annotations.get('return', sig.return_annotation)
    rt = self.__class__.parse_annotation(anno, None, none_as_other=False)
    params_str = ''.join((x.code for x in res.values()))
    assert_or_throw(re.match(params_re, params_str) is not None, lambda: TypeError(f'Input types not valid {res} for {func}'))
    assert_or_throw(re.match(return_re, rt.code) is not None, lambda: TypeError(f'Return type not valid {rt} for {func}'))
    return (class_method, res, rt)