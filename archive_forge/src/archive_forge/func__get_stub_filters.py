from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
def _get_stub_filters(self, origin_scope):
    return [StubFilter(parent_context=self.as_context(), origin_scope=origin_scope)] + list(self.iter_star_filters())