import tempfile
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.functional import base
def test_stack_validation(self):
    test_template = tempfile.NamedTemporaryFile(delete=False)
    test_template.write(validate_template.encode('utf-8'))
    test_template.close()
    stack_name = self.getUniqueString('validate_template')
    self.assertRaises(exceptions.SDKException, self.user_cloud.create_stack, name=stack_name, template_file=test_template.name)