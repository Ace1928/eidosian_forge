from libcloud.test import MockHttp
from libcloud.utils.py3 import xmlrpclib
def _xmlrpc(self, method, url, body, headers):
    params, methodName = xmlrpclib.loads(body)
    meth_name = '_xmlrpc__' + methodName.replace('.', '_')
    if self.type:
        meth_name = '{}_{}'.format(meth_name, self.type)
    return getattr(self, meth_name)(method, url, body, headers)