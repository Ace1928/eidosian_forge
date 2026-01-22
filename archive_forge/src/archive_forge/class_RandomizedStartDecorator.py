import jinja2
from minerl.herobraine.hero.handler import Handler
class RandomizedStartDecorator(Handler):

    def to_string(self) -> str:
        return 'randomized_start_decorator'

    def xml_template(self) -> str:
        return str('<RandomizedStartDecorator/>')