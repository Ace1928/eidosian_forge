import collections
def _get_internal_key(self, key):
    """Return keys used for the internal dictionary."""
    if is_train(key):
        return KerasModeKeys.TRAIN
    if is_eval(key):
        return KerasModeKeys.TEST
    if is_predict(key):
        return KerasModeKeys.PREDICT
    raise ValueError('Invalid mode key: {}.'.format(key))