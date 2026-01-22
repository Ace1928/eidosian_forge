from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter, ScatterPlane
from kivy.properties import ObjectProperty
class ScatterLayout(Scatter):
    """ScatterLayout class, see module documentation for more information.
    """
    content = ObjectProperty()

    def __init__(self, **kw):
        self.content = FloatLayout()
        super(ScatterLayout, self).__init__(**kw)
        if self.content.size != self.size:
            self.content.size = self.size
        super(ScatterLayout, self).add_widget(self.content)
        self.fbind('size', self.update_size)

    def update_size(self, instance, size):
        self.content.size = size

    def add_widget(self, *args, **kwargs):
        self.content.add_widget(*args, **kwargs)

    def remove_widget(self, *args, **kwargs):
        self.content.remove_widget(*args, **kwargs)

    def clear_widgets(self, *args, **kwargs):
        self.content.clear_widgets(*args, **kwargs)