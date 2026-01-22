import string
import textwrap
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib import parse as urlparse
def _windows_callback_script(passwd=None):
    script_url = 'https://raw.githubusercontent.com/ansible/ansible/devel/examples/scripts/ConfigureRemotingForAnsible.ps1'
    if passwd is not None:
        passwd = passwd.replace("'", "''")
        script_tpl = "        <powershell>\n        $admin = [adsi]('WinNT://./administrator, user')\n        $admin.PSBase.Invoke('SetPassword', '${PASS}')\n        Invoke-Expression ((New-Object System.Net.Webclient).DownloadString('${SCRIPT}'))\n        </powershell>\n        "
    else:
        script_tpl = "        <powershell>\n        $admin = [adsi]('WinNT://./administrator, user')\n        Invoke-Expression ((New-Object System.Net.Webclient).DownloadString('${SCRIPT}'))\n        </powershell>\n        "
    tpl = string.Template(textwrap.dedent(script_tpl))
    return tpl.safe_substitute(PASS=passwd, SCRIPT=script_url)