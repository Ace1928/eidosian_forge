import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Enum, Instance, Required
from ..model import Model
@abstract
class Marking(Model):
    """ Base class for graphical markings, e.g. arrow heads.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)