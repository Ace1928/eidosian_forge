from kivy.uix.floatlayout import FloatLayout
class RelativeLayout(FloatLayout):
    """RelativeLayout class, see module documentation for more information.
    """

    def __init__(self, **kw):
        super(RelativeLayout, self).__init__(**kw)
        funbind = self.funbind
        trigger = self._trigger_layout
        funbind('pos', trigger)
        funbind('pos_hint', trigger)

    def do_layout(self, *args):
        super(RelativeLayout, self).do_layout(pos=(0, 0))

    def to_parent(self, x, y, **k):
        return (x + self.x, y + self.y)

    def to_local(self, x, y, **k):
        return (x - self.x, y - self.y)

    def _apply_transform(self, m, pos=None):
        m.translate(self.x, self.y, 0)
        return super(RelativeLayout, self)._apply_transform(m, (0, 0))

    def on_motion(self, etype, me):
        if me.type_id in self.motion_filter and 'pos' in me.profile:
            me.push()
            me.apply_transform_2d(self.to_local)
            ret = super().on_motion(etype, me)
            me.pop()
            return ret
        return super().on_motion(etype, me)

    def on_touch_down(self, touch):
        x, y = (touch.x, touch.y)
        touch.push()
        touch.apply_transform_2d(self.to_local)
        ret = super(RelativeLayout, self).on_touch_down(touch)
        touch.pop()
        return ret

    def on_touch_move(self, touch):
        x, y = (touch.x, touch.y)
        touch.push()
        touch.apply_transform_2d(self.to_local)
        ret = super(RelativeLayout, self).on_touch_move(touch)
        touch.pop()
        return ret

    def on_touch_up(self, touch):
        x, y = (touch.x, touch.y)
        touch.push()
        touch.apply_transform_2d(self.to_local)
        ret = super(RelativeLayout, self).on_touch_up(touch)
        touch.pop()
        return ret