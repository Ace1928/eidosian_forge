import datetime
import math
import typing as t
from wandb.util import (
class UnionType(Type):
    """An "or" of types."""
    name = 'union'
    types: t.ClassVar[t.List[type]] = []

    def __init__(self, allowed_types: t.Optional[t.Sequence[ConvertableToType]]=None):
        assert allowed_types is None or allowed_types.__class__ == list
        if allowed_types is None:
            wb_types = []
        else:
            wb_types = [TypeRegistry.type_from_dtype(dt) for dt in allowed_types]
        wb_types = _flatten_union_types(wb_types)
        wb_types.sort(key=str)
        self.params.update({'allowed_types': wb_types})

    def assign(self, py_obj: t.Optional[t.Any]=None) -> t.Union['UnionType', InvalidType]:
        resolved_types = _union_assigner(self.params['allowed_types'], py_obj, type_mode=False)
        if isinstance(resolved_types, InvalidType):
            return InvalidType()
        return self.__class__(resolved_types)

    def assign_type(self, wb_type: 'Type') -> t.Union['UnionType', InvalidType]:
        if isinstance(wb_type, UnionType):
            assignees = wb_type.params['allowed_types']
        else:
            assignees = [wb_type]
        resolved_types = self.params['allowed_types']
        for assignee in assignees:
            resolved_types = _union_assigner(resolved_types, assignee, type_mode=True)
            if isinstance(resolved_types, InvalidType):
                return InvalidType()
        return self.__class__(resolved_types)

    def explain(self, other: t.Any, depth=0) -> str:
        exp = super().explain(other, depth)
        for ndx, subtype in enumerate(self.params['allowed_types']):
            if ndx > 0:
                exp += '\n{}and'.format(''.join(['\t'] * depth))
            exp += '\n' + subtype.explain(other, depth=depth + 1)
        return exp

    def __repr__(self):
        return '{}'.format(' or '.join([str(t) for t in self.params['allowed_types']]))