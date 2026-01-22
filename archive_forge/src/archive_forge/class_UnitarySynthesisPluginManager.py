import abc
from typing import List
import stevedore
class UnitarySynthesisPluginManager:
    """Unitary Synthesis plugin manager class

    This class tracks the installed plugins, it has a single property,
    ``ext_plugins`` which contains a list of stevedore plugin objects.
    """

    def __init__(self):
        self.ext_plugins = stevedore.ExtensionManager('qiskit.unitary_synthesis', invoke_on_load=True, propagate_map_exceptions=True)