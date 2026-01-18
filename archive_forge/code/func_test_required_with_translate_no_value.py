from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
@mock.patch.object(translation.Translation, 'has_translation')
@mock.patch.object(translation.Translation, 'translate')
def test_required_with_translate_no_value(self, m_t, m_ht):
    m_t.return_value = None
    m_ht.return_value = True
    self.assertRaises(ValueError, self.props.get, 'required_int')