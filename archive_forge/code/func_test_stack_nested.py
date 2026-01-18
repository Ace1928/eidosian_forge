import tempfile
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.functional import base
def test_stack_nested(self):
    test_template = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    test_template.write(root_template.encode('utf-8'))
    test_template.close()
    simple_tmpl = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    simple_tmpl.write(fakes.FAKE_TEMPLATE.encode('utf-8'))
    simple_tmpl.close()
    env = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    expanded_env = environment % simple_tmpl.name
    env.write(expanded_env.encode('utf-8'))
    env.close()
    self.stack_name = self.getUniqueString('nested_stack')
    self.addCleanup(self._cleanup_stack)
    stack = self.user_cloud.create_stack(name=self.stack_name, template_file=test_template.name, environment_files=[env.name], wait=True)
    self.assertEqual('CREATE_COMPLETE', stack['stack_status'])
    rands = stack['outputs'][0]['output_value']
    self.assertEqual(['0', '1', '2', '3', '4'], sorted(rands.keys()))
    for rand in rands.values():
        self.assertEqual(10, len(rand))