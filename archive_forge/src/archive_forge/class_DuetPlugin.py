from typing import Callable, List, Optional
from mypy.plugin import FunctionContext, Plugin
from mypy.types import CallableType, get_proper_type, Instance, Overloaded, Type
class DuetPlugin(Plugin):

    def get_function_hook(self, fullname: str) -> Optional[Callable[[FunctionContext], Type]]:
        if fullname == 'duet.api.sync':
            return duet_sync_callback
        return None