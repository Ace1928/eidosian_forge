from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
class AgentQuitFromTouchingBlockType(Handler):

    def to_string(self) -> str:
        return 'agent_quit_from_touching_block_type'

    def xml_template(self) -> str:
        return str('<AgentQuitFromTouchingBlockType>\n                    {% for block in blocks %}\n                    <Block type="{{ block }}"/>\n                    {% endfor %}\n                </AgentQuitFromTouchingBlockType>')

    def __init__(self, blocks: List[str]):
        """Creates a reward which will cause the player to quit when they touch a block."""
        self.blocks = blocks