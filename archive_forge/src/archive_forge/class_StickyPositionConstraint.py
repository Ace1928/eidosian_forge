from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
@dataclass
class StickyPositionConstraint:
    """
    Sticky position constraints.
    """
    sticky_box_rect: dom.Rect
    containing_block_rect: dom.Rect
    nearest_layer_shifting_sticky_box: typing.Optional[LayerId] = None
    nearest_layer_shifting_containing_block: typing.Optional[LayerId] = None

    def to_json(self):
        json = dict()
        json['stickyBoxRect'] = self.sticky_box_rect.to_json()
        json['containingBlockRect'] = self.containing_block_rect.to_json()
        if self.nearest_layer_shifting_sticky_box is not None:
            json['nearestLayerShiftingStickyBox'] = self.nearest_layer_shifting_sticky_box.to_json()
        if self.nearest_layer_shifting_containing_block is not None:
            json['nearestLayerShiftingContainingBlock'] = self.nearest_layer_shifting_containing_block.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(sticky_box_rect=dom.Rect.from_json(json['stickyBoxRect']), containing_block_rect=dom.Rect.from_json(json['containingBlockRect']), nearest_layer_shifting_sticky_box=LayerId.from_json(json['nearestLayerShiftingStickyBox']) if 'nearestLayerShiftingStickyBox' in json else None, nearest_layer_shifting_containing_block=LayerId.from_json(json['nearestLayerShiftingContainingBlock']) if 'nearestLayerShiftingContainingBlock' in json else None)