from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class AgentStartVelocity(Handler):

    def to_string(self) -> str:
        return f'agent_start_velocity({self.x}, {self.y}, {self.z})'

    def xml_template(self) -> str:
        return str('<Velocity x="{{x}}" y="{{y}}" z="{{z}}"/>')

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z