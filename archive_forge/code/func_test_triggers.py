from collections import defaultdict
import pytest
from modin.config import Parameter
def test_triggers(prefilled_parameter):
    results = defaultdict(int)
    callbacks = []

    def make_callback(name, res=results):

        def callback(p: Parameter):
            res[name] += 1
        callbacks.append(callback)
        return callback
    prefilled_parameter.once('init', make_callback('init'))
    assert results['init'] == 1
    prefilled_parameter.once('never', make_callback('never'))
    prefilled_parameter.once('once', make_callback('once'))
    prefilled_parameter.subscribe(make_callback('subscribe'))
    prefilled_parameter.put('multi')
    prefilled_parameter.put('once')
    prefilled_parameter.put('multi')
    prefilled_parameter.put('once')
    expected = [('init', 1), ('never', 0), ('once', 1), ('subscribe', 5)]
    for name, val in expected:
        assert results[name] == val, '{} has wrong count'.format(name)