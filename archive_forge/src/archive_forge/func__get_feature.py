def _get_feature(name):
    import __future__
    return getattr(__future__, name, object())