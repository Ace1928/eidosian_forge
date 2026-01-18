from typing import Optional
from minerl.herobraine.hero.handlers.agent.action import Action, ItemListAction
import jinja2
import minerl.herobraine.hero.spaces as spaces

        Initializes the space of the handler to be one for each item in the list
        Requires 0th item to be 'none' and last item to be 'other' coresponding to
        no-op and non-listed item respectively
        