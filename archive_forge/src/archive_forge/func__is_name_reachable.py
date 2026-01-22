from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
def _is_name_reachable(self, name):
    if not super()._is_name_reachable(name):
        return False
    definition = name.get_definition()
    if definition is None:
        return False
    if definition.type in ('import_from', 'import_name'):
        if name.parent.type not in ('import_as_name', 'dotted_as_name'):
            return False
    n = name.value
    if n.startswith('_') and (not (n.startswith('__') and n.endswith('__'))):
        return False
    return True