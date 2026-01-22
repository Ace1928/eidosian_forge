import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
class MappingEngineTester(BaseApp):
    """Execute mapping engine locally."""
    name = 'mapping_engine'

    def __init__(self):
        super(MappingEngineTester, self).__init__()
        self.mapping_id = uuid.uuid4().hex
        self.rules_pathname = None
        self.rules = None
        self.assertion_pathname = None
        self.assertion = None

    def read_rules(self, path):
        self.rules_pathname = path
        try:
            with open(path, 'rb') as file:
                self.rules = jsonutils.load(file)
        except ValueError as e:
            raise SystemExit(_('Error while parsing rules %(path)s: %(err)s') % {'path': path, 'err': e})

    def read_assertion(self, path):
        self.assertion_pathname = path
        try:
            with open(path) as file:
                self.assertion = file.read().strip()
        except IOError as e:
            raise SystemExit(_('Error while opening file %(path)s: %(err)s') % {'path': path, 'err': e})
        LOG.debug('Assertions loaded: [%s].', self.assertion)

    def normalize_assertion(self):

        def split(line, line_num):
            try:
                k, v = line.split(':', 1)
                return (k.strip(), v.strip())
            except ValueError:
                msg = _("assertion file %(pathname)s at line %(line_num)d expected 'key: value' but found '%(line)s' see help for file format")
                raise SystemExit(msg % {'pathname': self.assertion_pathname, 'line_num': line_num, 'line': line})
        assertion = self.assertion.splitlines()
        assertion_dict = {}
        prefix = CONF.command.prefix
        for line_num, line in enumerate(assertion, 1):
            line = line.strip()
            if line == '':
                continue
            k, v = split(line, line_num)
            if prefix:
                if k.startswith(prefix):
                    assertion_dict[k] = v
            else:
                assertion_dict[k] = v
        self.assertion = assertion_dict

    def normalize_rules(self):
        if isinstance(self.rules, list):
            self.rules = {'rules': self.rules}

    @classmethod
    def main(cls):
        if CONF.command.engine_debug:
            mapping_engine.LOG.logger.setLevel('DEBUG')
            LOG.logger.setLevel('DEBUG')
            LOG.debug('Debug log level enabled!')
        else:
            mapping_engine.LOG.logger.setLevel('WARN')
        tester = cls()
        tester.read_rules(CONF.command.rules)
        tester.normalize_rules()
        attribute_mapping = tester.rules.copy()
        if CONF.command.mapping_schema_version:
            attribute_mapping['schema_version'] = CONF.command.mapping_schema_version
        if not attribute_mapping.get('schema_version'):
            default_schema_version = '1.0'
            LOG.warning('No schema version defined in rules [%s]. Therefore,we will use the default as [%s].', attribute_mapping, default_schema_version)
            attribute_mapping['schema_version'] = default_schema_version
        LOG.info('Validating Attribute mapping rules [%s].', attribute_mapping)
        mapping_engine.validate_mapping_structure(attribute_mapping)
        LOG.info('Attribute mapping rules are valid.')
        tester.read_assertion(CONF.command.input)
        tester.normalize_assertion()
        if CONF.command.engine_debug:
            print('Using Rules:\n%s' % jsonutils.dumps(tester.rules, indent=2))
            print('Using Assertion:\n%s' % jsonutils.dumps(tester.assertion, indent=2))
        rp = mapping_engine.RuleProcessor(tester.mapping_id, tester.rules['rules'])
        mapped = rp.process(tester.assertion)
        LOG.info('Result of the attribute mapping processing.')
        print(jsonutils.dumps(mapped, indent=2))

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(MappingEngineTester, cls).add_argument_parser(subparsers)
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.add_argument('--rules', default=None, required=True, help="Path to the file with rules to be executed. Content must be\na proper JSON structure, with a top-level key 'rules' and\ncorresponding value being a list.")
        parser.add_argument('--input', default=None, required=True, help="Path to the file with input attributes. The content\nconsists of ':' separated parameter names and their values.\nThere is only one key-value pair per line. A ';' in the\nvalue is a separator and then a value is treated as a list.\nExample:\n\tEMAIL: me@example.com\n\tLOGIN: me\n\tGROUPS: group1;group2;group3")
        parser.add_argument('--prefix', default=None, help='A prefix used for each environment variable in the\nassertion. For example, all environment variables may have\nthe prefix ASDF_.')
        parser.add_argument('--engine-debug', default=False, action='store_true', help='Enable debug messages from the mapping engine.')
        parser.add_argument('--mapping-schema-version', default=None, required=False, help="The override for the schema version of the rules that are loaded in the 'rules' option of the test CLI.")