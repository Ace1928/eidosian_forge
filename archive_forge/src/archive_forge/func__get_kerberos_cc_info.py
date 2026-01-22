import importlib
import importlib.metadata
import typing as t
import traceback
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _get_kerberos_cc_info(self, ctx: 'krb5.Context') -> t.Dict[str, t.Any]:
    creds: t.List[t.Dict[str, t.Any]] = []
    res: t.Dict[str, t.Any] = {'exception': None, 'name': None, 'principal': None, 'creds': creds}
    try:
        default_cc = krb5.cc_default(ctx)
    except Exception:
        res['exception'] = traceback.format_exc()
        return res
    try:
        res['name'] = str(default_cc)
        res['principal'] = str(default_cc.principal)
        for cred in default_cc:
            creds.append({'client': str(getattr(cred, 'client', 'krb5 too old')), 'server': str(getattr(cred, 'server', 'krb5 too old'))})
    except Exception:
        res['exception'] = traceback.format_exc()
    return res