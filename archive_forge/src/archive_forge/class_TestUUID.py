import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class TestUUID(TestField):

    def setUp(self):
        super(TestUUID, self).setUp()
        self.field = fields.UUIDField()
        self.coerce_good_values = [('da66a411-af0e-4829-9b67-475017ddd152', 'da66a411-af0e-4829-9b67-475017ddd152'), ('da66a411af0e48299b67475017ddd152', 'da66a411af0e48299b67475017ddd152'), ('DA66A411-AF0E-4829-9B67-475017DDD152', 'DA66A411-AF0E-4829-9B67-475017DDD152'), ('DA66A411AF0E48299b67475017DDD152', 'DA66A411AF0E48299b67475017DDD152'), ('da66a411-af0e-4829-9b67', 'da66a411-af0e-4829-9b67'), ('da66a411-af0e-4829-9b67-475017ddd152548999', 'da66a411-af0e-4829-9b67-475017ddd152548999'), ('da66a411-af0e-4829-9b67-475017ddz152', 'da66a411-af0e-4829-9b67-475017ddz152'), ('fake_uuid', 'fake_uuid'), ('fake_uāid', 'fake_uāid'), (b'fake_u\xe1id'.decode('latin_1'), b'fake_u\xe1id'.decode('latin_1')), ('1', '1'), (1, '1')]
        self.to_primitive_values = self.coerce_good_values[0:1]
        self.from_primitive_values = self.coerce_good_values[0:1]

    @mock.patch('warnings.warn')
    def test_coerce_good_values(self, mock_warn):
        super().test_coerce_good_values()
        mock_warn.assert_has_calls([mock.call(mock.ANY, FutureWarning)] * 8)

    def test_validation_warning_can_be_escalated_to_exception(self):
        warnings.filterwarnings(action='error')
        self.assertRaises(FutureWarning, self.field.coerce, 'obj', 'attr', 'not a uuid')

    def test_get_schema(self):
        field = fields.UUIDField()
        schema = field.get_schema()
        self.assertEqual(['string'], schema['type'])
        self.assertEqual(False, schema['readonly'])
        pattern = schema['pattern']
        for _, valid_val in self.coerce_good_values[:4]:
            self.assertRegex(valid_val, pattern)
        invalid_vals = [x for x in self.coerce_bad_values if isinstance(x, str)]
        for invalid_val in invalid_vals:
            self.assertNotRegex(invalid_val, pattern)