from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def transform_with_touch(self, touch):
    changed = False
    if len(self._touches) == self.translation_touches:
        dx = (touch.x - self._last_touch_pos[touch][0]) * self.do_translation_x
        dy = (touch.y - self._last_touch_pos[touch][1]) * self.do_translation_y
        dx = dx / self.translation_touches
        dy = dy / self.translation_touches
        self.apply_transform(Matrix().translate(dx, dy, 0))
        changed = True
    if len(self._touches) == 1:
        return changed
    points = [Vector(self._last_touch_pos[t]) for t in self._touches if t is not touch]
    points.append(Vector(touch.pos))
    anchor = max(points[:-1], key=lambda p: p.distance(touch.pos))
    farthest = max(points, key=anchor.distance)
    if farthest is not points[-1]:
        return changed
    old_line = Vector(*touch.ppos) - anchor
    new_line = Vector(*touch.pos) - anchor
    if not old_line.length():
        return changed
    angle = radians(new_line.angle(old_line)) * self.do_rotation
    if angle:
        changed = True
    self.apply_transform(Matrix().rotate(angle, 0, 0, 1), anchor=anchor)
    if self.do_scale:
        scale = new_line.length() / old_line.length()
        new_scale = scale * self.scale
        if new_scale < self.scale_min:
            scale = self.scale_min / self.scale
        elif new_scale > self.scale_max:
            scale = self.scale_max / self.scale
        self.apply_transform(Matrix().scale(scale, scale, scale), anchor=anchor)
        changed = True
    return changed