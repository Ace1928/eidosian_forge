from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
@staticmethod
def _draw_dirty_internal(_old_rect, _rect, _sprites, _surf_blit, _update, _special_flags):
    for spr in _sprites:
        flags = spr.blendmode if _special_flags is None else _special_flags
        if spr.dirty < 1 and spr.visible:
            if spr.source_rect is not None:
                _spr_rect = _rect(spr.rect.topleft, spr.source_rect.size)
                rect_offset_x = spr.source_rect[0] - _spr_rect[0]
                rect_offset_y = spr.source_rect[1] - _spr_rect[1]
            else:
                _spr_rect = spr.rect
                rect_offset_x = -_spr_rect[0]
                rect_offset_y = -_spr_rect[1]
            _spr_rect_clip = _spr_rect.clip
            for idx in _spr_rect.collidelistall(_update):
                clip = _spr_rect_clip(_update[idx])
                _surf_blit(spr.image, clip, (clip[0] + rect_offset_x, clip[1] + rect_offset_y, clip[2], clip[3]), flags)
        else:
            if spr.visible:
                _old_rect[spr] = _surf_blit(spr.image, spr.rect, spr.source_rect, flags)
            if spr.dirty == 1:
                spr.dirty = 0