import jinja2
from minerl.herobraine.hero.handler import Handler
class DrawingDecorator(Handler):

    def __init__(self, to_draw: str):
        self.to_draw = to_draw

    def xml_template(self) -> str:
        tmp = '<DrawingDecorator>{{to_draw}}</DrawingDecorator>'
        return tmp

    def to_string(self) -> str:
        return 'drawing_decorator'