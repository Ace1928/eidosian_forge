import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def two_way_scenarios():
    scenarios = [('PP', {'make_delta': _groupcompress_py.make_delta, 'apply_delta': _groupcompress_py.apply_delta})]
    if compiled_groupcompress_feature.available():
        gc_module = compiled_groupcompress_feature.module
        scenarios.extend([('CC', {'make_delta': gc_module.make_delta, 'apply_delta': gc_module.apply_delta}), ('PC', {'make_delta': _groupcompress_py.make_delta, 'apply_delta': gc_module.apply_delta}), ('CP', {'make_delta': gc_module.make_delta, 'apply_delta': _groupcompress_py.apply_delta})])
    return scenarios