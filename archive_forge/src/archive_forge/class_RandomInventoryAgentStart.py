from minerl.herobraine.hero.handler import Handler
from typing import Dict, List, Union
import random
import jinja2
class RandomInventoryAgentStart(InventoryAgentStart):
    """ An inventory agentstart specification which
    that fills
    """

    def __init__(self, inventory: Dict[str, Union[str, int]], use_hotbar: bool=False):
        """ Creates an inventory where items are placed in random positions

        For example:

            rias =  RandomInventoryAgentStart({'dirt': 10, 'planks': 5})
        """
        self.inventory = inventory
        self.slot_range = (0, 36) if use_hotbar else (10, 36)

    def xml_template(self) -> str:
        lines = ['<Inventory>']
        for item, quantity in self.inventory.items():
            slot = random.randint(*self.slot_range)
            lines.append(f'<InventoryObject slot="{slot}" type="{item}" quantity="{quantity}"/>')
        lines.append('</Inventory>')
        return '\n'.join(lines)