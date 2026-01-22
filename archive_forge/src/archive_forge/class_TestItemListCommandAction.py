from minerl.herobraine.hero.handlers.agent.observations.compass import CompassObservation
from minerl.herobraine.hero.handlers.agent.observations.inventory import FlatInventoryObservation
from minerl.herobraine.hero.handlers.agent.observations.equipped_item import _TypeObservation
from minerl.herobraine.hero.handlers.agent.action import ItemListAction
class TestItemListCommandAction(ItemListAction):

    def __init__(self, items: list, **item_list_kwargs):
        super().__init__('test', items, **item_list_kwargs)

    def to_string(self):
        return 'test_item_list_command'