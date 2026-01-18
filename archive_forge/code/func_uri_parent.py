from .schema import rest_translation
def uri_parent(uri):
    files_index = uri.find('/files/')
    if files_index >= 0:
        uri = uri[:7 + files_index]
    return uri_split(uri)[0]