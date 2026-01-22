from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSRule:
    """
    CSS rule representation.
    """
    selector_list: SelectorList
    origin: StyleSheetOrigin
    style: CSSStyle
    style_sheet_id: typing.Optional[StyleSheetId] = None
    nesting_selectors: typing.Optional[typing.List[str]] = None
    media: typing.Optional[typing.List[CSSMedia]] = None
    container_queries: typing.Optional[typing.List[CSSContainerQuery]] = None
    supports: typing.Optional[typing.List[CSSSupports]] = None
    layers: typing.Optional[typing.List[CSSLayer]] = None
    scopes: typing.Optional[typing.List[CSSScope]] = None
    rule_types: typing.Optional[typing.List[CSSRuleType]] = None

    def to_json(self):
        json = dict()
        json['selectorList'] = self.selector_list.to_json()
        json['origin'] = self.origin.to_json()
        json['style'] = self.style.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        if self.nesting_selectors is not None:
            json['nestingSelectors'] = [i for i in self.nesting_selectors]
        if self.media is not None:
            json['media'] = [i.to_json() for i in self.media]
        if self.container_queries is not None:
            json['containerQueries'] = [i.to_json() for i in self.container_queries]
        if self.supports is not None:
            json['supports'] = [i.to_json() for i in self.supports]
        if self.layers is not None:
            json['layers'] = [i.to_json() for i in self.layers]
        if self.scopes is not None:
            json['scopes'] = [i.to_json() for i in self.scopes]
        if self.rule_types is not None:
            json['ruleTypes'] = [i.to_json() for i in self.rule_types]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(selector_list=SelectorList.from_json(json['selectorList']), origin=StyleSheetOrigin.from_json(json['origin']), style=CSSStyle.from_json(json['style']), style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None, nesting_selectors=[str(i) for i in json['nestingSelectors']] if 'nestingSelectors' in json else None, media=[CSSMedia.from_json(i) for i in json['media']] if 'media' in json else None, container_queries=[CSSContainerQuery.from_json(i) for i in json['containerQueries']] if 'containerQueries' in json else None, supports=[CSSSupports.from_json(i) for i in json['supports']] if 'supports' in json else None, layers=[CSSLayer.from_json(i) for i in json['layers']] if 'layers' in json else None, scopes=[CSSScope.from_json(i) for i in json['scopes']] if 'scopes' in json else None, rule_types=[CSSRuleType.from_json(i) for i in json['ruleTypes']] if 'ruleTypes' in json else None)