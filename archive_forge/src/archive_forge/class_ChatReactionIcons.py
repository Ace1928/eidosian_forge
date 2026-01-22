from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..layout import Column
from ..reactive import ReactiveHTML
from ..widgets.base import CompositeWidget
from ..widgets.icon import ToggleIcon
class ChatReactionIcons(CompositeWidget):
    """
    A widget to display reaction icons that can be clicked on.

    Parameters
    ----------
    value : List
        The selected reactions.
    options : Dict
        A key-value pair of reaction values and their corresponding tabler icon names
        found on https://tabler-icons.io.
    active_icons : Dict
        The mapping of reactions to their corresponding active icon names;
        if not set, the active icon name will default to its "filled" version.

    Reference: https://panel.holoviz.org/reference/chat/ChatReactionIcons.html

    :Example:

    >>> ChatReactionIcons(value=["like"], options={"like": "thumb-up", "dislike": "thumb-down"})
    """
    active_icons = param.Dict(default={}, doc='\n        The mapping of reactions to their corresponding active icon names.\n        If not set, the active icon name will default to its "filled" version.')
    css_classes = param.List(default=['reaction-icons'], doc='The CSS classes of the widget.')
    options = param.Dict(default={'favorite': 'heart'}, doc='\n        A key-value pair of reaction values and their corresponding tabler icon names\n        found on https://tabler-icons.io.')
    value = param.List(doc='The active reactions.')
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/chat_reaction_icons.css']
    _composite_type = Column

    def __init__(self, **params):
        super().__init__(**params)
        self._render_icons()

    @param.depends('options', watch=True)
    def _render_icons(self):
        self._rendered_icons = {}
        for option, icon in self.options.items():
            active_icon = self.active_icons.get(option, '')
            icon = ToggleIcon(icon=icon, active_icon=active_icon, value=option in self.value, margin=0)
            icon._reaction = option
            icon.param.watch(self._update_value, 'value')
            self._rendered_icons[option] = icon
        self._composite[:] = list(self._rendered_icons.values())

    @param.depends('value', watch=True)
    def _update_icons(self):
        for option, icon in self._rendered_icons.items():
            icon.value = option in self.value

    @param.depends('active_icons', watch=True)
    def _update_active_icons(self):
        for option, icon in self._rendered_icons.items():
            icon.active_icon = self.active_icons.get(option, '')

    def _update_value(self, event):
        reaction = event.obj._reaction
        icon_value = event.new
        reactions = self.value.copy()
        if icon_value and reaction not in self.value:
            reactions.append(reaction)
        elif not icon_value and reaction in self.value:
            reactions.remove(reaction)
        self.value = reactions