import json
from typing import Callable, Tuple, cast
from adagio.exceptions import (DependencyDefinitionError,
from adagio.instances import _ConfigVar, _Input, _Output, _Task
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import (ConfigSpec, InputSpec, OutputSpec, TaskSpec,
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid
def test_taskspec():
    configs = [dict(name='ca', data_type=int, nullable=False, required=False, default_value=2)]
    inputs = [dict(name='ia', data_type=str, nullable=True, required=True, timeout='1s'), dict(name='ib', data_type=str, nullable=True, required=True, timeout='1s')]
    outputs = [dict(name='oa', data_type=float, nullable=False)]
    func = _mock_task_func
    metadata = dict(x=1, y='b')
    ts = TaskSpec(configs, inputs, outputs, func, metadata)
    j = ts.to_json(True)
    j2 = TaskSpec(**json.loads(j)).to_json(True)
    assert j == j2
    j = ts.to_json(False)
    j2 = TaskSpec(**json.loads(j)).to_json(False)
    assert j == j2
    configs = [ConfigSpec(**configs[0])]
    outputs = [json.dumps(OutputSpec(**outputs[0]).jsondict)]
    ts = TaskSpec(configs, inputs, outputs, func, metadata)
    j2 = ts.to_json(False)
    assert j == j2
    assert to_taskspec(j2).__uuid__() == ts.__uuid__()