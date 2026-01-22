import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Enum, Instance, Required
from ..model import Model
class Decoration(Model):
    """ Indicates a positioned marker, e.g. at a node of a glyph.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    marking = Instance(Marking, help='\n    The graphical marking associated with this decoration, e.g. an arrow head.\n    ')
    node = Required(Enum('start', 'middle', 'end'), help='\n    The placement of the marking on the parent graphical object.\n    ')