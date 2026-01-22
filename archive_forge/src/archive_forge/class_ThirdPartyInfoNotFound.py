class ThirdPartyInfoNotFound(Exception):
    """ Raised from implementation of ThirdPartyLocator.third_party_info when
    the info cannot be found.
    """
    pass