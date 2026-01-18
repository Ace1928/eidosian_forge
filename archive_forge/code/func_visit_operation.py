from typing import Dict, Tuple, Union
from langchain.chains.query_constructor.ir import (
def visit_operation(self, operation: Operation) -> Dict:
    args = [arg.accept(self) for arg in operation.arguments]
    return {self._format_func(operation.operator): args}