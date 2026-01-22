import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
class URISourceTestCase(base.BaseTestCase):

    def setUp(self):
        super(URISourceTestCase, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.conf_fixture = self.useFixture(fixture.Config(self.conf))

    def _register_opts(self, opts):
        for g in opts.keys():
            for o, (t, _) in opts[g].items():
                self.conf.register_opt(t(o), g if g != 'DEFAULT' else None)

    def test_incomplete_driver(self):
        self.conf_fixture.load_raw_values(group='incomplete_ini_driver', driver='remote_file')
        source = self.conf._open_source_from_opt_group('incomplete_ini_driver')
        self.assertIsNone(source)

    @requests_mock.mock()
    def test_fetch_uri(self, m):
        m.get('https://bad.uri', status_code=404)
        self.assertRaises(HTTPError, _uri.URIConfigurationSource, 'https://bad.uri')
        m.get('https://good.uri', text='[DEFAULT]\nfoo=bar\n')
        source = _uri.URIConfigurationSource('https://good.uri')
        self.assertEqual('bar', source.get('DEFAULT', 'foo', cfg.StrOpt('foo'))[0])

    @base.mock.patch('oslo_config.sources._uri.URIConfigurationSource._fetch_uri', side_effect=opts_to_ini)
    def test_configuration_source(self, mock_fetch_uri):
        group = 'types'
        uri = make_uri(group)
        self.conf_fixture.load_raw_values(group=group, driver='remote_file', uri=uri)
        self.conf_fixture.config(config_source=[group])
        self.assertEqual(self.conf._sources, [])
        self.conf._load_alternative_sources()
        self.assertIsInstance(self.conf._sources[0], _uri.URIConfigurationSource)
        source = self.conf._open_source_from_opt_group(group)
        self._register_opts(_extra_configs[uri]['data'])
        self.assertIs(sources._NoValue, source.get('DEFAULT', 'bar', cfg.StrOpt('bar'))[0])
        for g in _extra_configs[uri]['data']:
            for o, (t, v) in _extra_configs[uri]['data'][g].items():
                self.assertEqual(str(v), str(source.get(g, o, t(o))[0]))
                self.assertEqual(v, self.conf[g][o] if g != 'DEFAULT' else self.conf[o])

    @base.mock.patch('oslo_config.sources._uri.URIConfigurationSource._fetch_uri', side_effect=opts_to_ini)
    def test_multiple_configuration_sources(self, mock_fetch_uri):
        groups = ['ini_1', 'ini_2', 'ini_3']
        uri = make_uri('ini_3')
        for group in groups:
            self.conf_fixture.load_raw_values(group=group, driver='remote_file', uri=make_uri(group))
        self.conf_fixture.config(config_source=groups)
        self.conf._load_alternative_sources()
        self._register_opts(_extra_configs[uri]['data'])
        for option in _extra_configs[uri]['data']['DEFAULT']:
            self.assertEqual(option, self.conf[option])

    def test_list_opts(self):
        discovered_group = None
        for group in _list_opts.list_opts():
            if group[0] is not None:
                if group[0].name == 'sample_remote_file_source':
                    discovered_group = group
                    break
        self.assertIsNotNone(discovered_group)
        self.assertEqual(_uri.URIConfigurationSourceDriver().list_options_for_discovery(), discovered_group[1][1:])