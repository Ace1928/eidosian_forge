import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
def _test_resolve_rule(self, is_list=False, check_error=False):

    class FakeClientPlugin(object):

        def find_name_id(self, entity=None, src_value='far'):
            if check_error:
                raise exception.NotFound()
            if entity == 'rose':
                return 'pink'
            return 'yellow'
    if is_list:
        schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING)}))}
    else:
        schema = {'far': properties.Schema(properties.Schema.STRING)}
    return (FakeClientPlugin(), schema)