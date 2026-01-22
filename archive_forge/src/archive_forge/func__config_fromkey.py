import sys
import dns._features
def _config_fromkey(self, key, always_try_domain):
    try:
        servers, _ = winreg.QueryValueEx(key, 'NameServer')
    except WindowsError:
        servers = None
    if servers:
        self._config_nameservers(servers)
    if servers or always_try_domain:
        try:
            dom, _ = winreg.QueryValueEx(key, 'Domain')
            if dom:
                self.info.domain = _config_domain(dom)
        except WindowsError:
            pass
    else:
        try:
            servers, _ = winreg.QueryValueEx(key, 'DhcpNameServer')
        except WindowsError:
            servers = None
        if servers:
            self._config_nameservers(servers)
            try:
                dom, _ = winreg.QueryValueEx(key, 'DhcpDomain')
                if dom:
                    self.info.domain = _config_domain(dom)
            except WindowsError:
                pass
    try:
        search, _ = winreg.QueryValueEx(key, 'SearchList')
    except WindowsError:
        search = None
    if search is None:
        try:
            search, _ = winreg.QueryValueEx(key, 'DhcpSearchList')
        except WindowsError:
            search = None
    if search:
        self._config_search(search)