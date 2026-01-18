from .schema import rest_translation
def uri_segment(uri, start=None, end=None):
    if start is None and end is None:
        return uri
    elif start is None:
        return '/' + '/'.join(uri.split('/')[:end])
    elif end is None:
        return '/' + '/'.join(uri.split('/')[start:])
    else:
        return '/' + '/'.join(uri.split('/')[start:end])