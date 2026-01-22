import sys, netrc
import curl
class SourceForgeUserSession(curl.Curl):

    def login(self, name, password):
        """Establish a login session."""
        self.post('account/login.php', (('form_loginname', name), ('form_pw', password), ('return_to', ''), ('stay_in_ssl', '1'), ('login', 'Login With SSL')))

    def logout(self):
        """Log out of SourceForge."""
        self.get('account/logout.php')

    def fetch_xml(self, numid):
        self.get('export/xml_export.php?group_id=%s' % numid)