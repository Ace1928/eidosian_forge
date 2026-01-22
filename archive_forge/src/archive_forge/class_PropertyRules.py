import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
class PropertyRules(object):

    def __init__(self, policy_enforcer=None):
        self.rules = []
        self.prop_exp_mapping = {}
        self.policies = []
        self.policy_enforcer = policy_enforcer or glance.api.policy.Enforcer()
        self.prop_prot_rule_format = CONF.property_protection_rule_format
        self.prop_prot_rule_format = self.prop_prot_rule_format.lower()
        self._load_rules()

    def _load_rules(self):
        try:
            conf_file = CONF.find_file(CONF.property_protection_file)
            CONFIG.read(conf_file)
        except Exception as e:
            msg = _LE("Couldn't find property protection file %(file)s: %(error)s.") % {'file': CONF.property_protection_file, 'error': e}
            LOG.error(msg)
            raise InvalidPropProtectConf()
        if self.prop_prot_rule_format not in ['policies', 'roles']:
            msg = _LE("Invalid value '%s' for 'property_protection_rule_format'. The permitted values are 'roles' and 'policies'") % self.prop_prot_rule_format
            LOG.error(msg)
            raise InvalidPropProtectConf()
        operations = ['create', 'read', 'update', 'delete']
        properties = CONFIG.sections()
        for property_exp in properties:
            property_dict = {}
            compiled_rule = self._compile_rule(property_exp)
            for operation in operations:
                try:
                    permissions = CONFIG.get(property_exp, operation)
                except configparser.NoOptionError:
                    raise InvalidPropProtectConf()
                if permissions:
                    if self.prop_prot_rule_format == 'policies':
                        if ',' in permissions:
                            LOG.error(_LE("Multiple policies '%s' not allowed for a given operation. Policies can be combined in the policy file"), permissions)
                            raise InvalidPropProtectConf()
                        self.prop_exp_mapping[compiled_rule] = property_exp
                        self._add_policy_rules(property_exp, operation, permissions)
                        permissions = [permissions]
                    else:
                        permissions = [permission.strip() for permission in permissions.split(',')]
                        if '@' in permissions and '!' in permissions:
                            msg = _LE("Malformed property protection rule in [%(prop)s] %(op)s=%(perm)s: '@' and '!' are mutually exclusive") % dict(prop=property_exp, op=operation, perm=permissions)
                            LOG.error(msg)
                            raise InvalidPropProtectConf()
                    property_dict[operation] = permissions
                else:
                    property_dict[operation] = []
                    LOG.warning(_LW('Property protection on operation %(operation)s for rule %(rule)s is not found. No role will be allowed to perform this operation.'), {'operation': operation, 'rule': property_exp})
            self.rules.append((compiled_rule, property_dict))

    def _compile_rule(self, rule):
        try:
            return re.compile(rule)
        except Exception as e:
            msg = _LE('Encountered a malformed property protection rule %(rule)s: %(error)s.') % {'rule': rule, 'error': e}
            LOG.error(msg)
            raise InvalidPropProtectConf()

    def _add_policy_rules(self, property_exp, action, rule):
        """Add policy rules to the policy enforcer.

        For example, if the file listed as property_protection_file has:
        [prop_a]
        create = glance_creator
        then the corresponding policy rule would be:
        "prop_a:create": "rule:glance_creator"
        where glance_creator is defined in policy.yaml. For example:
        "glance_creator": "role:admin or role:glance_create_user"
        """
        rule = 'rule:%s' % rule
        rule_name = '%s:%s' % (property_exp, action)
        rule_dict = policy.Rules.from_dict({rule_name: rule})
        self.policy_enforcer.add_rules(rule_dict)

    def _check_policy(self, property_exp, action, context):
        try:
            action = ':'.join([property_exp, action])
            self.policy_enforcer.enforce(context, action, {}, registered=False)
        except exception.Forbidden:
            return False
        return True

    def check_property_rules(self, property_name, action, context):
        roles = context.roles
        if context.service_roles:
            roles.extend(context.service_roles)
        if not self.rules:
            return True
        if action not in ['create', 'read', 'update', 'delete']:
            return False
        for rule_exp, rule in self.rules:
            if rule_exp.search(str(property_name)):
                break
        else:
            return False
        rule_roles = rule.get(action)
        if rule_roles:
            if '!' in rule_roles:
                return False
            elif '@' in rule_roles:
                return True
            if self.prop_prot_rule_format == 'policies':
                prop_exp_key = self.prop_exp_mapping[rule_exp]
                return self._check_policy(prop_exp_key, action, context)
            if set(roles).intersection(set([role.lower() for role in rule_roles])):
                return True
        return False