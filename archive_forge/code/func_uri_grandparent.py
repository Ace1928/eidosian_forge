from .schema import rest_translation
def uri_grandparent(uri):
    return uri_parent(uri_parent(uri))