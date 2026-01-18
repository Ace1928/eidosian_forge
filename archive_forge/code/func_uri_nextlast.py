from .schema import rest_translation
def uri_nextlast(uri):
    if '/files/' in uri:
        return 'files'
    return uri.split('/')[-2]