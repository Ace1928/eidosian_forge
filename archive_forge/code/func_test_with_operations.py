import textwrap
from oslotest import base
from oslo_policy import policy
from oslo_policy import sphinxext
def test_with_operations(self):
    results = '\n'.join(list(sphinxext._format_policy_section('foo', [policy.DocumentedRuleDefault('rule_a', '@', 'My sample rule', [{'method': 'GET', 'path': '/foo'}, {'method': 'POST', 'path': '/some'}])])))
    self.assertEqual(textwrap.dedent('\n        foo\n        ===\n\n        ``rule_a``\n            :Default: ``@``\n            :Operations:\n                - **GET** ``/foo``\n                - **POST** ``/some``\n\n            My sample rule\n        ').lstrip(), results)