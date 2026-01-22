from mistralclient.api import base
def _get_workbooks_url(self, resource=None, namespace=None, scope=None):
    url = '/workbooks'
    if resource:
        url += '/%s' % resource
    if scope and namespace:
        url += '?scope=%s&namespace=%s' % (scope, namespace)
    elif scope:
        url += '?scope=%s' % scope
    elif namespace:
        url += '?namespace=%s' % namespace
    return url