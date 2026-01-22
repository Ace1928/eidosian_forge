from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class LayeredDirty(LayeredUpdates):
    """LayeredDirty Group is for DirtySprites; subclasses LayeredUpdates

    pygame.sprite.LayeredDirty(*sprites, **kwargs): return LayeredDirty

    This group requires pygame.sprite.DirtySprite or any sprite that
    has the following attributes:
        image, rect, dirty, visible, blendmode (see doc of DirtySprite).

    It uses the dirty flag technique and is therefore faster than
    pygame.sprite.RenderUpdates if you have many static sprites.  It
    also switches automatically between dirty rect updating and full
    screen drawing, so you do no have to worry which would be faster.

    As with the pygame.sprite.Group, you can specify some additional attributes
    through kwargs:
        _use_update: True/False   (default is False)
        _default_layer: default layer where the sprites without a layer are
            added
        _time_threshold: threshold time for switching between dirty rect mode
            and fullscreen mode; defaults to updating at 80 frames per second,
            which is equal to 1000.0 / 80.0

    New in pygame 1.8.0

    """

    def __init__(self, *sprites, **kwargs):
        """initialize group.

        pygame.sprite.LayeredDirty(*sprites, **kwargs): return LayeredDirty

        You can specify some additional attributes through kwargs:
            _use_update: True/False   (default is False)
            _default_layer: default layer where the sprites without a layer are
                added
            _time_threshold: threshold time for switching between dirty rect
                mode and fullscreen mode; defaults to updating at 80 frames per
                second, which is equal to 1000.0 / 80.0

        """
        LayeredUpdates.__init__(self, *sprites, **kwargs)
        self._clip = None
        self._use_update = False
        self._time_threshold = 1000.0 / 80.0
        self._bgd = None
        for key, val in kwargs.items():
            if key in ['_use_update', '_time_threshold', '_default_layer'] and hasattr(self, key):
                setattr(self, key, val)

    def add_internal(self, sprite, layer=None):
        """Do not use this method directly.

        It is used by the group to add a sprite internally.

        """
        if not hasattr(sprite, 'dirty'):
            raise AttributeError()
        if not hasattr(sprite, 'visible'):
            raise AttributeError()
        if not hasattr(sprite, 'blendmode'):
            raise AttributeError()
        if not isinstance(sprite, DirtySprite):
            raise TypeError()
        if sprite.dirty == 0:
            sprite.dirty = 1
        LayeredUpdates.add_internal(self, sprite, layer)

    def draw(self, surface, bgsurf=None, special_flags=None):
        """draw all sprites in the right order onto the given surface

        LayeredDirty.draw(surface, bgsurf=None, special_flags=None): return Rect_list

        You can pass the background too. If a self.bgd is already set to some
        value that is not None, then the bgsurf argument has no effect.
        Passing a value to special_flags will pass that value as the
        special_flags argument to pass to all Surface.blit calls, overriding
        the sprite.blendmode attribute

        """
        orig_clip = surface.get_clip()
        latest_clip = self._clip
        if latest_clip is None:
            latest_clip = orig_clip
        local_sprites = self._spritelist
        local_old_rect = self.spritedict
        local_update = self.lostsprites
        rect_type = Rect
        surf_blit_func = surface.blit
        if bgsurf is not None:
            self._bgd = bgsurf
        local_bgd = self._bgd
        surface.set_clip(latest_clip)
        start_time = get_ticks()
        if self._use_update:
            self._find_dirty_area(latest_clip, local_old_rect, rect_type, local_sprites, local_update, local_update.append, self._init_rect)
            if local_bgd is not None:
                flags = 0 if special_flags is None else special_flags
                for rec in local_update:
                    surf_blit_func(local_bgd, rec, rec, flags)
            self._draw_dirty_internal(local_old_rect, rect_type, local_sprites, surf_blit_func, local_update, special_flags)
            local_ret = list(local_update)
        else:
            if local_bgd is not None:
                flags = 0 if special_flags is None else special_flags
                surf_blit_func(local_bgd, (0, 0), None, flags)
            for spr in local_sprites:
                if spr.visible:
                    flags = spr.blendmode if special_flags is None else special_flags
                    local_old_rect[spr] = surf_blit_func(spr.image, spr.rect, spr.source_rect, flags)
            local_ret = [rect_type(latest_clip)]
        end_time = get_ticks()
        if end_time - start_time > self._time_threshold:
            self._use_update = False
        else:
            self._use_update = True
        local_update[:] = []
        surface.set_clip(orig_clip)
        return local_ret

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

    @staticmethod
    def _find_dirty_area(_clip, _old_rect, _rect, _sprites, _update, _update_append, init_rect):
        for spr in _sprites:
            if spr.dirty > 0:
                if spr.source_rect:
                    _union_rect = _rect(spr.rect.topleft, spr.source_rect.size)
                else:
                    _union_rect = _rect(spr.rect)
                _union_rect_collidelist = _union_rect.collidelist
                _union_rect_union_ip = _union_rect.union_ip
                i = _union_rect_collidelist(_update)
                while i > -1:
                    _union_rect_union_ip(_update[i])
                    del _update[i]
                    i = _union_rect_collidelist(_update)
                _update_append(_union_rect.clip(_clip))
                if _old_rect[spr] is not init_rect:
                    _union_rect = _rect(_old_rect[spr])
                    _union_rect_collidelist = _union_rect.collidelist
                    _union_rect_union_ip = _union_rect.union_ip
                    i = _union_rect_collidelist(_update)
                    while i > -1:
                        _union_rect_union_ip(_update[i])
                        del _update[i]
                        i = _union_rect_collidelist(_update)
                    _update_append(_union_rect.clip(_clip))

    def clear(self, surface, bgd):
        """use to set background

        Group.clear(surface, bgd): return None

        """
        self._bgd = bgd

    def repaint_rect(self, screen_rect):
        """repaint the given area

        LayeredDirty.repaint_rect(screen_rect): return None

        screen_rect is in screen coordinates.

        """
        if self._clip:
            self.lostsprites.append(screen_rect.clip(self._clip))
        else:
            self.lostsprites.append(Rect(screen_rect))

    def set_clip(self, screen_rect=None):
        """clip the area where to draw; pass None (default) to reset the clip

        LayeredDirty.set_clip(screen_rect=None): return None

        """
        if screen_rect is None:
            self._clip = pygame.display.get_surface().get_rect()
        else:
            self._clip = screen_rect
        self._use_update = False

    def get_clip(self):
        """get the area where drawing will occur

        LayeredDirty.get_clip(): return Rect

        """
        return self._clip

    def change_layer(self, sprite, new_layer):
        """change the layer of the sprite

        LayeredUpdates.change_layer(sprite, new_layer): return None

        The sprite must have been added to the renderer already. This is not
        checked.

        """
        LayeredUpdates.change_layer(self, sprite, new_layer)
        if sprite.dirty == 0:
            sprite.dirty = 1

    def set_timing_treshold(self, time_ms):
        """set the threshold in milliseconds

        set_timing_treshold(time_ms): return None

        Defaults to 1000.0 / 80.0. This means that the screen will be painted
        using the flip method rather than the update method if the update
        method is taking so long to update the screen that the frame rate falls
        below 80 frames per second.

        Raises TypeError if time_ms is not int or float.

        """
        warn('This function will be removed, use set_timing_threshold function instead', DeprecationWarning)
        self.set_timing_threshold(time_ms)

    def set_timing_threshold(self, time_ms):
        """set the threshold in milliseconds

        set_timing_threshold(time_ms): return None

        Defaults to 1000.0 / 80.0. This means that the screen will be painted
        using the flip method rather than the update method if the update
        method is taking so long to update the screen that the frame rate falls
        below 80 frames per second.

        Raises TypeError if time_ms is not int or float.

        """
        if isinstance(time_ms, (int, float)):
            self._time_threshold = time_ms
        else:
            raise TypeError(f'Expected numeric value, got {time_ms.__class__.__name__} instead')