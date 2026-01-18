import json
from typing import Callable, Tuple, cast
from adagio.exceptions import (DependencyDefinitionError,
from adagio.instances import _ConfigVar, _Input, _Output, _Task
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import (ConfigSpec, InputSpec, OutputSpec, TaskSpec,
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid
def test_workflowspec():
    configs = [dict(name='cx', data_type=str, nullable=False, required=False, default_value='c')]
    inputs = [dict(name='ia', data_type=int, nullable=False), dict(name='ib', data_type=str, nullable=True)]
    outputs = [dict(name='oa', data_type=str, nullable=False), dict(name='ob', data_type=str, nullable=True)]

    def is_config(ds):
        return [d['data_type'] is not int for d in ds]

    def is_int(ds):
        return [d['data_type'] is int for d in ds]
    f = WorkflowSpec(configs, inputs, outputs, {})
    f.add_task('a', function_to_taskspec(f0, is_config), [])
    f.add_task('b', function_to_taskspec(f1, is_config), dependency=dict(a='a._0'), config=dict(b='bb'))
    f.add_task('c', function_to_taskspec(f1, is_config), dependency=dict(a='a._1'), config=dict(b='bc'))
    raises(KeyError, lambda: f.add_task('b', function_to_taskspec(f1, is_config), dependency=dict(a='a._0'), config=dict(b='bb')))
    raises(DependencyDefinitionError, lambda: f.add_task('d', function_to_taskspec(f1, is_config), dependency=dict(a='a._0')))
    raises(DependencyDefinitionError, lambda: f.add_task('d', function_to_taskspec(f1, is_config), dependency=dict(a='a._0'), config=dict(b='bb'), config_dependency=dict(b='cx')))
    raises(DependencyNotDefinedError, lambda: f.add_task('d', function_to_taskspec(f1, is_config), dependency={'a.b': 'a._0'}, config=dict(b='bb')))
    raises(DependencyDefinitionError, lambda: f.add_task('d', function_to_taskspec(f1, is_config), dependency={'a': 'a.b._0'}, config=dict(b='bb')))
    raises(DependencyDefinitionError, lambda: f.add_task('d', function_to_taskspec(f1, is_int), dependency={'b': 'a._0'}, config=dict(a='bb')))
    f.add_task('d', function_to_taskspec(f2, is_config), dependency={'a': 'b._0', 'b': 'c._0'}, config_dependency=dict(c='cx'))
    f.add_task('e', function_to_taskspec(f1, is_config), dependency=dict(a='ia'), config=dict(b='bb'))
    raises(DependencyNotDefinedError, lambda: f.validate())
    raises(DependencyDefinitionError, lambda: f.link('oa', 'b._0'))
    f.link('oa', 'd._0')
    raises(DependencyNotDefinedError, lambda: f.validate())
    f.link('ob', 'ib')
    f.validate()
    j1 = f.to_json(False)
    f_ = to_taskspec(j1)
    j2 = f_.to_json(False)
    assert j1 == j2
    assert f.__uuid__() == f_.__uuid__()