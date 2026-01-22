from kivy.uix.layout import Layout
class FloatLayout(Layout):
    """Float layout class. See module documentation for more information.
    """

    def __init__(self, **kwargs):
        super(FloatLayout, self).__init__(**kwargs)
        fbind = self.fbind
        update = self._trigger_layout
        fbind('children', update)
        fbind('pos', update)
        fbind('pos_hint', update)
        fbind('size_hint', update)
        fbind('size', update)

    def do_layout(self, *largs, **kwargs):
        w, h = kwargs.get('size', self.size)
        x, y = kwargs.get('pos', self.pos)
        for c in self.children:
            shw, shh = c.size_hint
            shw_min, shh_min = c.size_hint_min
            shw_max, shh_max = c.size_hint_max
            if shw is not None and shh is not None:
                c_w = shw * w
                c_h = shh * h
                if shw_min is not None and c_w < shw_min:
                    c_w = shw_min
                elif shw_max is not None and c_w > shw_max:
                    c_w = shw_max
                if shh_min is not None and c_h < shh_min:
                    c_h = shh_min
                elif shh_max is not None and c_h > shh_max:
                    c_h = shh_max
                c.size = (c_w, c_h)
            elif shw is not None:
                c_w = shw * w
                if shw_min is not None and c_w < shw_min:
                    c_w = shw_min
                elif shw_max is not None and c_w > shw_max:
                    c_w = shw_max
                c.width = c_w
            elif shh is not None:
                c_h = shh * h
                if shh_min is not None and c_h < shh_min:
                    c_h = shh_min
                elif shh_max is not None and c_h > shh_max:
                    c_h = shh_max
                c.height = c_h
            for key, value in c.pos_hint.items():
                if key == 'x':
                    c.x = x + value * w
                elif key == 'right':
                    c.right = x + value * w
                elif key == 'pos':
                    c.pos = (x + value[0] * w, y + value[1] * h)
                elif key == 'y':
                    c.y = y + value * h
                elif key == 'top':
                    c.top = y + value * h
                elif key == 'center':
                    c.center = (x + value[0] * w, y + value[1] * h)
                elif key == 'center_x':
                    c.center_x = x + value * w
                elif key == 'center_y':
                    c.center_y = y + value * h

    def add_widget(self, widget, *args, **kwargs):
        widget.bind(pos=self._trigger_layout, pos_hint=self._trigger_layout)
        return super(FloatLayout, self).add_widget(widget, *args, **kwargs)

    def remove_widget(self, widget, *args, **kwargs):
        widget.unbind(pos=self._trigger_layout, pos_hint=self._trigger_layout)
        return super(FloatLayout, self).remove_widget(widget, *args, **kwargs)