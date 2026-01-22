from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class DragData:
    items: typing.List[DragDataItem]
    drag_operations_mask: int
    files: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['items'] = [i.to_json() for i in self.items]
        json['dragOperationsMask'] = self.drag_operations_mask
        if self.files is not None:
            json['files'] = [i for i in self.files]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(items=[DragDataItem.from_json(i) for i in json['items']], drag_operations_mask=int(json['dragOperationsMask']), files=[str(i) for i in json['files']] if 'files' in json else None)