import json
from typing import Callable, Tuple, cast
from adagio.exceptions import (DependencyDefinitionError,
from adagio.instances import _ConfigVar, _Input, _Output, _Task
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import (ConfigSpec, InputSpec, OutputSpec, TaskSpec,
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid
def test_outputspec():
    raises(AssertionError, lambda: OutputSpec('', int, True))
    raises(TypeError, lambda: OutputSpec('a', 'xyz', True))
    o = OutputSpec('a', _Task, True)
    x = MockTaskForVar()
    assert x is o.validate_value(x)
    assert o.validate_value(None) is None
    o = OutputSpec('a', _Task, False)
    x = MockTaskForVar()
    assert x is o.validate_value(x)
    raises(AssertionError, lambda: o.validate_value(None))