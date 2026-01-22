from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
class RpmKey(object):

    def __init__(self, module):
        keyfile = None
        should_cleanup_keyfile = False
        self.module = module
        self.rpm = self.module.get_bin_path('rpm', True)
        state = module.params['state']
        key = module.params['key']
        fingerprint = module.params['fingerprint']
        if fingerprint:
            fingerprint = fingerprint.replace(' ', '').upper()
        self.gpg = self.module.get_bin_path('gpg')
        if not self.gpg:
            self.gpg = self.module.get_bin_path('gpg2', required=True)
        if '://' in key:
            keyfile = self.fetch_key(key)
            keyid = self.getkeyid(keyfile)
            should_cleanup_keyfile = True
        elif self.is_keyid(key):
            keyid = key
        elif os.path.isfile(key):
            keyfile = key
            keyid = self.getkeyid(keyfile)
        else:
            self.module.fail_json(msg='Not a valid key %s' % key)
        keyid = self.normalize_keyid(keyid)
        if state == 'present':
            if self.is_key_imported(keyid):
                module.exit_json(changed=False)
            else:
                if not keyfile:
                    self.module.fail_json(msg='When importing a key, a valid file must be given')
                if fingerprint:
                    has_fingerprint = self.getfingerprint(keyfile)
                    if fingerprint != has_fingerprint:
                        self.module.fail_json(msg="The specified fingerprint, '%s', does not match the key fingerprint '%s'" % (fingerprint, has_fingerprint))
                self.import_key(keyfile)
                if should_cleanup_keyfile:
                    self.module.cleanup(keyfile)
                module.exit_json(changed=True)
        elif self.is_key_imported(keyid):
            self.drop_key(keyid)
            module.exit_json(changed=True)
        else:
            module.exit_json(changed=False)

    def fetch_key(self, url):
        """Downloads a key from url, returns a valid path to a gpg key"""
        rsp, info = fetch_url(self.module, url)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch key at %s , error was: %s' % (url, info['msg']))
        key = rsp.read()
        if not is_pubkey(key):
            self.module.fail_json(msg='Not a public key: %s' % url)
        tmpfd, tmpname = tempfile.mkstemp()
        self.module.add_cleanup_file(tmpname)
        tmpfile = os.fdopen(tmpfd, 'w+b')
        tmpfile.write(key)
        tmpfile.close()
        return tmpname

    def normalize_keyid(self, keyid):
        """Ensure a keyid doesn't have a leading 0x, has leading or trailing whitespace, and make sure is uppercase"""
        ret = keyid.strip().upper()
        if ret.startswith('0x'):
            return ret[2:]
        elif ret.startswith('0X'):
            return ret[2:]
        else:
            return ret

    def getkeyid(self, keyfile):
        stdout, stderr = self.execute_command([self.gpg, '--no-tty', '--batch', '--with-colons', '--fixed-list-mode', keyfile])
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('pub:'):
                return line.split(':')[4]
        self.module.fail_json(msg='Unexpected gpg output')

    def getfingerprint(self, keyfile):
        stdout, stderr = self.execute_command([self.gpg, '--no-tty', '--batch', '--with-colons', '--fixed-list-mode', '--with-fingerprint', keyfile])
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('fpr:'):
                return line.split(':')[9]
        self.module.fail_json(msg='Unexpected gpg output')

    def is_keyid(self, keystr):
        """Verifies if a key, as provided by the user is a keyid"""
        return re.match('(0x)?[0-9a-f]{8}', keystr, flags=re.IGNORECASE)

    def execute_command(self, cmd):
        rc, stdout, stderr = self.module.run_command(cmd, use_unsafe_shell=True)
        if rc != 0:
            self.module.fail_json(msg=stderr)
        return (stdout, stderr)

    def is_key_imported(self, keyid):
        cmd = self.rpm + ' -q  gpg-pubkey'
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            return False
        cmd += ' --qf "%{description}" | ' + self.gpg + ' --no-tty --batch --with-colons --fixed-list-mode -'
        stdout, stderr = self.execute_command(cmd)
        for line in stdout.splitlines():
            if keyid in line.split(':')[4]:
                return True
        return False

    def import_key(self, keyfile):
        if not self.module.check_mode:
            self.execute_command([self.rpm, '--import', keyfile])

    def drop_key(self, keyid):
        if not self.module.check_mode:
            self.execute_command([self.rpm, '--erase', '--allmatches', 'gpg-pubkey-%s' % keyid[-8:].lower()])