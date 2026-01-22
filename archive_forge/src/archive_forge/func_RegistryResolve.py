import re
import winreg
def RegistryResolve():
    nameservers = []
    x = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        y = winreg.OpenKey(x, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters')
    except EnvironmentError:
        try:
            y = winreg.OpenKey(x, 'SYSTEM\\CurrentControlSet\\Services\\VxD\\MSTCP')
            nameserver, dummytype = winreg.QueryValueEx(y, 'NameServer')
            if nameserver and (not nameserver in nameservers):
                nameservers.extend(stringdisplay(nameserver))
        except EnvironmentError:
            pass
        return nameservers
    try:
        nameserver = winreg.QueryValueEx(y, 'DhcpNameServer')[0].split()
    except:
        nameserver = winreg.QueryValueEx(y, 'NameServer')[0].split()
    if nameserver:
        nameservers = nameserver
    nameserver = winreg.QueryValueEx(y, 'NameServer')[0]
    winreg.CloseKey(y)
    try:
        y = winreg.OpenKey(x, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\DNSRegisteredAdapters')
        for i in range(1000):
            try:
                n = winreg.EnumKey(y, i)
                z = winreg.OpenKey(y, n)
                dnscount, dnscounttype = winreg.QueryValueEx(z, 'DNSServerAddressCount')
                dnsvalues, dnsvaluestype = winreg.QueryValueEx(z, 'DNSServerAddresses')
                nameservers.extend(binipdisplay(dnsvalues))
                winreg.CloseKey(z)
            except EnvironmentError:
                break
        winreg.CloseKey(y)
    except EnvironmentError:
        pass
    try:
        y = winreg.OpenKey(x, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces')
        for i in range(1000):
            try:
                n = winreg.EnumKey(y, i)
                z = winreg.OpenKey(y, n)
                try:
                    nameserver, dummytype = winreg.QueryValueEx(z, 'NameServer')
                    if nameserver and (not nameserver in nameservers):
                        nameservers.extend(stringdisplay(nameserver))
                except EnvironmentError:
                    pass
                winreg.CloseKey(z)
            except EnvironmentError:
                break
        winreg.CloseKey(y)
    except EnvironmentError:
        pass
    winreg.CloseKey(x)
    return nameservers