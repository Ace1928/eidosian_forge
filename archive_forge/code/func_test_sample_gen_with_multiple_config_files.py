from unittest import mock
from oslotest import base
from oslo_policy import sphinxpolicygen
@mock.patch('os.path.isdir')
@mock.patch('os.path.isfile')
@mock.patch('oslo_policy.generator.generate_sample')
def test_sample_gen_with_multiple_config_files(self, sample, isfile, isdir):
    isfile.side_effect = [False, True] * 2
    isdir.return_value = True
    config = mock.Mock(policy_generator_config_file=[('nova.conf', 'nova'), ('placement.conf', 'placement')], exclude_deprecated=False)
    app = mock.Mock(srcdir='/opt/nova', config=config)
    sphinxpolicygen.generate_sample(app)
    sample.assert_has_calls([mock.call(args=['--config-file', '/opt/nova/nova.conf', '--output-file', '/opt/nova/nova.policy.yaml.sample'], conf=mock.ANY), mock.call(args=['--config-file', '/opt/nova/placement.conf', '--output-file', '/opt/nova/placement.policy.yaml.sample'], conf=mock.ANY)])