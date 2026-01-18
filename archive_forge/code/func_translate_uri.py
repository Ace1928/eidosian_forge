from .schema import rest_translation
def translate_uri(uri):
    segs = uri.split('/')
    for key in rest_translation.keys():
        if key in segs[-2:]:
            uri = uri.replace(key, rest_translation[key])
    return uri