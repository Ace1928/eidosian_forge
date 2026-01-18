from minerl.herobraine.hero.handlers.agent.actions.craft import CraftAction
from minerl.herobraine.hero.handlers.agent.action import Action, ItemListAction
import jinja2
import minerl.herobraine.hero.spaces as spaces

    An action handler for crafting items when agent is in view of a crafting table

        Note when used along side Craft Item, block lists must be disjoint or from_universal will fire multiple times

    