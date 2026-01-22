from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ConsoleMessage:
    """
    Console message.
    """
    source: str
    level: str
    text: str
    url: typing.Optional[str] = None
    line: typing.Optional[int] = None
    column: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['source'] = self.source
        json['level'] = self.level
        json['text'] = self.text
        if self.url is not None:
            json['url'] = self.url
        if self.line is not None:
            json['line'] = self.line
        if self.column is not None:
            json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(source=str(json['source']), level=str(json['level']), text=str(json['text']), url=str(json['url']) if 'url' in json else None, line=int(json['line']) if 'line' in json else None, column=int(json['column']) if 'column' in json else None)