from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class PgHba(object):
    """
    PgHba object to read/write entries to/from.
    pg_hba_file - the pg_hba file almost always /etc/pg_hba
    """

    def __init__(self, pg_hba_file=None, backup=False, create=False, keep_comments_at_rules=False):
        self.pg_hba_file = pg_hba_file
        self.rules = None
        self.comment = None
        self.backup = backup
        self.last_backup = None
        self.create = create
        self.keep_comments_at_rules = keep_comments_at_rules
        self.unchanged()
        self.databases = set(['postgres', 'template0', 'template1'])
        self.users = set(['postgres'])
        self.preexisting_rules = None
        self.read()

    def clear_rules(self):
        self.rules = {}

    def unchanged(self):
        """
        This method resets self.diff to a empty default
        """
        self.diff = {'before': {'file': self.pg_hba_file, 'pg_hba': []}, 'after': {'file': self.pg_hba_file, 'pg_hba': []}}

    def read(self):
        """
        Read in the pg_hba from the system
        """
        self.rules = {}
        self.comment = []
        try:
            with open(self.pg_hba_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    comment = None
                    if '#' in line:
                        line, comment = line.split('#', 1)
                        if comment == '':
                            comment = None
                        line = line.rstrip()
                    if line == '':
                        if comment is not None:
                            self.comment.append('#' + comment)
                    else:
                        if comment is not None and (not self.keep_comments_at_rules):
                            self.comment.append('#' + comment)
                            comment = None
                        try:
                            self.add_rule(PgHbaRule(line=line, comment=comment))
                        except PgHbaRuleError:
                            pass
            self.unchanged()
            self.preexisting_rules = dict(self.rules)
        except IOError:
            pass

    def write(self, backup_file=''):
        """
        This method writes the PgHba rules (back) to a file.
        """
        if not self.changed():
            return False
        contents = self.render()
        if self.pg_hba_file:
            if not (os.path.isfile(self.pg_hba_file) or self.create):
                raise PgHbaError("pg_hba file '{0}' doesn't exist. Use create option to autocreate.".format(self.pg_hba_file))
            if self.backup and os.path.isfile(self.pg_hba_file):
                if backup_file:
                    self.last_backup = backup_file
                else:
                    _backup_file_h, self.last_backup = tempfile.mkstemp(prefix='pg_hba')
                shutil.copy(self.pg_hba_file, self.last_backup)
            fileh = open(self.pg_hba_file, 'w')
        else:
            filed, _path = tempfile.mkstemp(prefix='pg_hba')
            fileh = os.fdopen(filed, 'w')
        fileh.write(contents)
        self.unchanged()
        fileh.close()
        return True

    def add_rule(self, rule):
        """
        This method can be used to add a rule to the list of rules in this PgHba object
        """
        key = rule.key()
        try:
            try:
                oldrule = self.rules[key]
            except KeyError:
                raise PgHbaRuleChanged
            ekeys = set(list(oldrule.keys()) + list(rule.keys()))
            ekeys.remove('line')
            for k in ekeys:
                if oldrule.get(k) != rule.get(k):
                    raise PgHbaRuleChanged('{0} changes {1}'.format(rule, oldrule))
        except PgHbaRuleChanged:
            self.rules[key] = rule
            self.diff['after']['pg_hba'].append(rule.line())
            if rule['db'] not in ['all', 'samerole', 'samegroup', 'replication']:
                databases = set(rule['db'].split(','))
                self.databases.update(databases)
            if rule['usr'] != 'all':
                user = rule['usr']
                if user[0] == '+':
                    user = user[1:]
                self.users.add(user)

    def remove_rule(self, rule):
        """
        This method can be used to find and remove a rule. It doesn't look for the exact rule, only
        the rule with the same key.
        """
        keys = rule.key()
        try:
            del self.rules[keys]
            self.diff['before']['pg_hba'].append(rule.line())
        except KeyError:
            pass

    def get_rules(self, with_lines=False):
        """
        This method returns all the rules of the PgHba object
        """
        rules = sorted(self.rules.values())
        for rule in rules:
            ret = {}
            for key, value in rule.items():
                ret[key] = value
            if not with_lines:
                if 'line' in ret:
                    del ret['line']
            else:
                ret['line'] = rule.line()
            yield ret

    def render(self):
        """
        This method renders the content of the PgHba rules and comments.
        The returning value can be used directly to write to a new file.
        """
        comment = '\n'.join(self.comment)
        rule_lines = []
        for rule in self.get_rules(with_lines=True):
            if 'comment' in rule:
                rule_lines.append(rule['line'] + '\t#' + rule['comment'])
            else:
                rule_lines.append(rule['line'])
        result = comment + '\n' + '\n'.join(rule_lines)
        if result and result[-1] not in ['\n', '\r']:
            result += '\n'
        return result

    def changed(self):
        """
        This method can be called to detect if the PgHba file has been changed.
        """
        if not self.preexisting_rules and (not self.rules):
            return False
        return self.preexisting_rules != self.rules