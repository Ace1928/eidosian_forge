import json
import six
def _loadfile(filename):
    try:
        with open(filename, 'r') as fp:
            obj = json.load(fp)
    except IOError as exc:
        raise InvalidClientSecretsError('Error opening file', exc.filename, exc.strerror, exc.errno)
    return _validate_clientsecrets(obj)