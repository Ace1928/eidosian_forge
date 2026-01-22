from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSStyleSheetHeader:
    """
    CSS stylesheet metainformation.
    """
    style_sheet_id: StyleSheetId
    frame_id: page.FrameId
    source_url: str
    origin: StyleSheetOrigin
    title: str
    disabled: bool
    is_inline: bool
    is_mutable: bool
    is_constructed: bool
    start_line: float
    start_column: float
    length: float
    end_line: float
    end_column: float
    source_map_url: typing.Optional[str] = None
    owner_node: typing.Optional[dom.BackendNodeId] = None
    has_source_url: typing.Optional[bool] = None
    loading_failed: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['styleSheetId'] = self.style_sheet_id.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['sourceURL'] = self.source_url
        json['origin'] = self.origin.to_json()
        json['title'] = self.title
        json['disabled'] = self.disabled
        json['isInline'] = self.is_inline
        json['isMutable'] = self.is_mutable
        json['isConstructed'] = self.is_constructed
        json['startLine'] = self.start_line
        json['startColumn'] = self.start_column
        json['length'] = self.length
        json['endLine'] = self.end_line
        json['endColumn'] = self.end_column
        if self.source_map_url is not None:
            json['sourceMapURL'] = self.source_map_url
        if self.owner_node is not None:
            json['ownerNode'] = self.owner_node.to_json()
        if self.has_source_url is not None:
            json['hasSourceURL'] = self.has_source_url
        if self.loading_failed is not None:
            json['loadingFailed'] = self.loading_failed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(style_sheet_id=StyleSheetId.from_json(json['styleSheetId']), frame_id=page.FrameId.from_json(json['frameId']), source_url=str(json['sourceURL']), origin=StyleSheetOrigin.from_json(json['origin']), title=str(json['title']), disabled=bool(json['disabled']), is_inline=bool(json['isInline']), is_mutable=bool(json['isMutable']), is_constructed=bool(json['isConstructed']), start_line=float(json['startLine']), start_column=float(json['startColumn']), length=float(json['length']), end_line=float(json['endLine']), end_column=float(json['endColumn']), source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None, owner_node=dom.BackendNodeId.from_json(json['ownerNode']) if 'ownerNode' in json else None, has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None, loading_failed=bool(json['loadingFailed']) if 'loadingFailed' in json else None)