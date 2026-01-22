import keras
def _multi_backend():
    version_fn = getattr(keras, 'version', None)
    return version_fn and version_fn().startswith('3.')