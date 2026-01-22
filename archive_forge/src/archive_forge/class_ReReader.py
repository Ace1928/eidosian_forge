import sys
from docutils import utils, parsers, Component
from docutils.transforms import universal
class ReReader(Reader):
    """
    A reader which rereads an existing document tree (e.g. a
    deserializer).

    Often used in conjunction with `writers.UnfilteredWriter`.
    """

    def get_transforms(self):
        return Component.get_transforms(self)