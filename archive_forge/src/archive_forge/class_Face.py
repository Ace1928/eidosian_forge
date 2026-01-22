from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class Face(Widget):
    """This draws a face with two eyes.

    It exposes a couple of properties
    to configure itself and hides all other details.
    """
    _attrMap = AttrMap(x=AttrMapValue(isNumber), y=AttrMapValue(isNumber), size=AttrMapValue(isNumber), skinColor=AttrMapValue(isColorOrNone), eyeColor=AttrMapValue(isColorOrNone), mood=AttrMapValue(OneOf('happy', 'sad', 'ok')))

    def __init__(self):
        self.x = 10
        self.y = 10
        self.size = 80
        self.skinColor = None
        self.eyeColor = colors.blue
        self.mood = 'happy'

    def demo(self):
        pass

    def draw(self):
        s = self.size
        g = shapes.Group()
        g.transform = [1, 0, 0, 1, self.x, self.y]
        g.add(shapes.Circle(s * 0.5, s * 0.5, s * 0.5, fillColor=self.skinColor))
        g.add(shapes.Circle(s * 0.35, s * 0.65, s * 0.1, fillColor=colors.white))
        g.add(shapes.Circle(s * 0.35, s * 0.65, s * 0.05, fillColor=self.eyeColor))
        g.add(shapes.Circle(s * 0.65, s * 0.65, s * 0.1, fillColor=colors.white))
        g.add(shapes.Circle(s * 0.65, s * 0.65, s * 0.05, fillColor=self.eyeColor))
        g.add(shapes.Polygon(points=[s * 0.5, s * 0.6, s * 0.4, s * 0.3, s * 0.6, s * 0.3], fillColor=None))
        if self.mood == 'happy':
            offset = -0.05
        elif self.mood == 'sad':
            offset = +0.05
        else:
            offset = 0
        g.add(shapes.Polygon(points=[s * 0.3, s * 0.2, s * 0.7, s * 0.2, s * 0.6, s * (0.2 + offset), s * 0.4, s * (0.2 + offset)], fillColor=colors.pink, strokeColor=colors.red, strokeWidth=s * 0.03))
        return g