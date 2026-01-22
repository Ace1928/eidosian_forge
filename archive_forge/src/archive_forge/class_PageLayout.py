from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, DictProperty
from kivy.animation import Animation
class PageLayout(Layout):
    """PageLayout class. See module documentation for more information.
    """
    page = NumericProperty(0)
    'The currently displayed page.\n\n    :data:`page` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to 0.\n    '
    border = NumericProperty('50dp')
    'The width of the border around the current page used to display\n    the previous/next page swipe areas when needed.\n\n    :data:`border` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 50dp.\n    '
    swipe_threshold = NumericProperty(0.5)
    'The threshold used to trigger swipes as ratio of the widget\n    size.\n\n    :data:`swipe_threshold` is a :class:`~kivy.properties.NumericProperty`\n    and defaults to .5.\n    '
    anim_kwargs = DictProperty({'d': 0.5, 't': 'in_quad'})
    "The animation kwargs used to construct the animation\n\n    :data:`anim_kwargs` is a :class:`~kivy.properties.DictProperty`\n    and defaults to {'d': .5, 't': 'in_quad'}.\n\n    .. versionadded:: 1.11.0\n    "

    def __init__(self, **kwargs):
        super(PageLayout, self).__init__(**kwargs)
        trigger = self._trigger_layout
        fbind = self.fbind
        fbind('border', trigger)
        fbind('page', trigger)
        fbind('parent', trigger)
        fbind('children', trigger)
        fbind('size', trigger)
        fbind('pos', trigger)

    def do_layout(self, *largs):
        l_children = len(self.children) - 1
        h = self.height
        x_parent, y_parent = self.pos
        p = self.page
        border = self.border
        half_border = border / 2.0
        right = self.right
        width = self.width - border
        for i, c in enumerate(reversed(self.children)):
            if i < p:
                x = x_parent
            elif i == p:
                if not p:
                    x = x_parent
                elif p != l_children:
                    x = x_parent + half_border
                else:
                    x = x_parent + border
            elif i == p + 1:
                if not p:
                    x = right - border
                else:
                    x = right - half_border
            else:
                x = right
            c.height = h
            c.width = width
            Animation(x=x, y=y_parent, **self.anim_kwargs).start(c)

    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos) or (not self.children):
            return
        page = self.children[-self.page - 1]
        if self.x <= touch.x < page.x:
            touch.ud['page'] = 'previous'
            touch.grab(self)
            return True
        elif page.right <= touch.x < self.right:
            touch.ud['page'] = 'next'
            touch.grab(self)
            return True
        return page.on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current != self:
            return
        p = self.page
        border = self.border
        half_border = border / 2.0
        page = self.children[-p - 1]
        if touch.ud['page'] == 'previous':
            if p < len(self.children) - 1:
                self.children[-p - 2].x = min(self.right - self.border * (1 - (touch.sx - touch.osx)), self.right)
            if p >= 1:
                b_right = half_border if p > 1 else border
                b_left = half_border if p < len(self.children) - 1 else border
                self.children[-p - 1].x = max(min(self.x + b_left + (touch.x - touch.ox), self.right - b_right), self.x + b_left)
            if p > 1:
                self.children[-p].x = min(self.x + half_border * (touch.sx - touch.osx), self.x + half_border)
        elif touch.ud['page'] == 'next':
            if p >= 1:
                self.children[-p - 1].x = max(self.x + half_border * (1 - (touch.osx - touch.sx)), self.x)
            if p < len(self.children) - 1:
                b_right = half_border if p >= 1 else border
                b_left = half_border if p < len(self.children) - 2 else border
                self.children[-p - 2].x = min(max(self.right - b_right + (touch.x - touch.ox), self.x + b_left), self.right - b_right)
            if p < len(self.children) - 2:
                self.children[-p - 3].x = max(self.right + half_border * (touch.sx - touch.osx), self.right - half_border)
        return page.on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current == self:
            if touch.ud['page'] == 'previous' and abs(touch.x - touch.ox) / self.width > self.swipe_threshold:
                self.page -= 1
            elif touch.ud['page'] == 'next' and abs(touch.x - touch.ox) / self.width > self.swipe_threshold:
                self.page += 1
            else:
                self._trigger_layout()
            touch.ungrab(self)
        if len(self.children) > 1:
            return self.children[-self.page + 1].on_touch_up(touch)