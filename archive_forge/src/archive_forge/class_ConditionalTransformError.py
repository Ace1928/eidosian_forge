from functools import wraps
from typing import Type
from pennylane import QueuingManager
from pennylane.operation import AnyWires, Operation, Operator
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
class ConditionalTransformError(ValueError):
    """Error for using qml.cond incorrectly"""