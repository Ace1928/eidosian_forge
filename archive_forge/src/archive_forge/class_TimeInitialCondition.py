from minerl.herobraine.hero.handler import Handler
import jinja2
class TimeInitialCondition(Handler):

    def to_string(self) -> str:
        return 'time_initial_condition'

    def xml_template(self) -> str:
        return str('<Time>\n                   {% if start_time is not none %}\n                   <StartTime>{{start_time | string}}</StartTime>\n                   {% endif %}\n                   <AllowPassageOfTime>{{allow_passage_of_time | string | lower}}</AllowPassageOfTime>\n                </Time>')

    def __init__(self, allow_passage_of_time: bool, start_time: int=None):
        self.start_time = start_time
        self.allow_passage_of_time = allow_passage_of_time