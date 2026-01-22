from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
import random
@register_script('display_model', aliases=['dm'])
class DisplayModel(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        display_model(self.opt)