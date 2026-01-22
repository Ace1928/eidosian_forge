from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
class StubModuleValue(ModuleValue):
    _module_name_class = StubModuleName

    def __init__(self, non_stub_value_set, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_stub_value_set = non_stub_value_set

    def is_stub(self):
        return True

    def sub_modules_dict(self):
        """
        We have to overwrite this, because it's possible to have stubs that
        don't have code for all the child modules. At the time of writing this
        there are for example no stubs for `json.tool`.
        """
        names = {}
        for value in self.non_stub_value_set:
            try:
                method = value.sub_modules_dict
            except AttributeError:
                pass
            else:
                names.update(method())
        names.update(super().sub_modules_dict())
        return names

    def _get_stub_filters(self, origin_scope):
        return [StubFilter(parent_context=self.as_context(), origin_scope=origin_scope)] + list(self.iter_star_filters())

    def get_filters(self, origin_scope=None):
        filters = super().get_filters(origin_scope)
        next(filters, None)
        stub_filters = self._get_stub_filters(origin_scope=origin_scope)
        yield from stub_filters
        yield from filters

    def _as_context(self):
        return StubModuleContext(self)