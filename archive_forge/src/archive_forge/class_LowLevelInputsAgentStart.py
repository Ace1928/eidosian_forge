from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class LowLevelInputsAgentStart(Handler):

    def to_string(self) -> str:
        return 'low_level_inputs'

    def xml_template(self) -> str:
        return '<LowLevelInputs>true</LowLevelInputs>'