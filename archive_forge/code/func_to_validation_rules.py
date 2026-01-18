from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def to_validation_rules(data: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for k, v in data.items():
        if k in ['partitionby_has', 'partitionby_is']:
            if isinstance(v, str):
                v = [x.strip() for x in v.split(',')]
            res[k] = PartitionSpec(by=v).partition_by
        elif k in ['presort_has', 'presort_is']:
            res[k] = list(parse_presort_exp(v).items())
        elif k in ['input_has']:
            if isinstance(v, str):
                res[k] = v.replace(' ', '').split(',')
            else:
                assert_or_throw(isinstance(v, list), lambda: SyntaxError(f'{v} is neither a string or a list'))
                res[k] = [x.replace(' ', '') for x in v]
        elif k in ['input_is']:
            try:
                res[k] = str(Schema(v))
            except SyntaxError:
                raise SyntaxError(f'for input_is, the input must be a schema expression {v}')
        else:
            raise NotImplementedError(k)
    return res